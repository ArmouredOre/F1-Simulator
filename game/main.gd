extends Node3D
## 3D client for server.py. The Python sim does all the physics; this draws it and sends your controls.
## Start `python server.py monza` first, then press Play (F5).
##
## Controller (F1-game layout): RT throttle, LT brake, left stick steer, Y/Triangle DRS (straight mode),
## B/Circle hold overtake, RB camera, right stick click look back, Start pause, Back/Select restart, X/Square guide.
## Keyboard: W/S or up/down throttle/brake, A/D or left/right steer, F DRS, Space overtake, C camera,
## B look back, P or Esc pause, R restart, G guide line on/off, [ and ] steering sensitivity.
## Back to track (after a spin or when stuck): LB or Backspace.
## Beginner assists (server default): left/right move the car across the track and the sim steers, brakes for
## corners and never damages the car. `python server.py monza --pro` turns them off.
## Sim axes: x east, y north, heading anticlockwise. Godot: (x, up, -y), so a heading is rotation.y unchanged.

const HOST := "127.0.0.1"
const PORT := 5555
# camera: [position behind/above the car in car axes (x forward, y up), pitch, how fast it swings behind the car]
const CAMS := [[Vector3(-7.5, 2.6, 0), -0.14, 6.0], [Vector3(-11, 4, 0), -0.2, 4.0], [Vector3(0.2, 0.95, 0), 0.0, 60.0]]
const CAR_COLORS := [Color(0.85, 0.1, 0.1), Color(0.1, 0.45, 0.95), Color(0.95, 0.75, 0.1)]
const MODEL_YAW := 0.0   # turn car.glb if it comes out of Blender facing the wrong way (it must face +x here)

var tcp := StreamPeerTCP.new()
var buf := ""
var connected := false
var retry := 0.0
var prev := {}   # the last two sim states, drawn blended so 50 Hz looks smooth at any frame rate
var cur := {}
var steer := 0.0
var thr := 0.0
var brk := 0.0
var drs := false
var sens := 1.0   # steering sensitivity, [ and ] change it while driving; put the value you like here to keep it
var paused := false
var want_reset := false
var want_rescue := false
var cam_i := 0
var last_dmg := 0.0

var cars: Array[Node3D] = []
var track: MeshInstance3D
var guide := MeshInstance3D.new()   # beginner guide: racing line, green = throttle, yellow = lift, red = brake
var rig := Node3D.new()
var cam := Camera3D.new()
var hud := {}
var map_line := Line2D.new()
var map_dots: Array[ColorRect] = []
var map_scale := 1.0
var map_mid := Vector2.ZERO


func _ready() -> void:
	var env := Environment.new()
	env.background_mode = Environment.BG_SKY
	env.sky = Sky.new()
	env.sky.sky_material = ProceduralSkyMaterial.new()
	env.ambient_light_sky_contribution = 0.5
	var we := WorldEnvironment.new()
	we.environment = env
	add_child(we)
	var sun := DirectionalLight3D.new()
	sun.rotation = Vector3(-0.9, 0.6, 0)
	sun.shadow_enabled = true
	sun.light_energy = 0.6
	add_child(sun)
	var grass := MeshInstance3D.new()
	grass.mesh = PlaneMesh.new()
	grass.mesh.size = Vector2(20000, 20000)
	grass.mesh.material = flat(Color(0.16, 0.36, 0.17))
	grass.position.y = -0.05
	add_child(grass)
	add_child(guide)
	add_child(rig)
	rig.add_child(cam)
	cam.far = 6000
	build_hud()


func flat(col: Color) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = col
	return m


# ---------- network ----------

func _physics_process(dt: float) -> void:
	tcp.poll()
	var status := tcp.get_status()
	if status == StreamPeerTCP.STATUS_CONNECTED:
		if not connected:
			connected = true
			tcp.set_no_delay(true)
		var n := tcp.get_available_bytes()
		if n > 0:
			buf += tcp.get_utf8_string(n)
		while "\n" in buf:
			var i := buf.find("\n")
			var msg: Variant = JSON.parse_string(buf.substr(0, i))
			buf = buf.substr(i + 1)
			if msg is Dictionary:
				if msg.has("track"):
					build_track(msg)
				else:
					prev = cur if not cur.is_empty() else msg
					cur = msg
		var c := controls(dt)
		if track and (not paused or want_reset):
			c["reset"] = want_reset
			c["rescue"] = want_rescue
			want_reset = false
			want_rescue = false
			tcp.put_data((JSON.stringify(c) + "\n").to_utf8_buffer())
	else:
		connected = false
		buf = ""
		retry -= dt
		if status != StreamPeerTCP.STATUS_CONNECTING and retry <= 0:
			retry = 1.0
			tcp.disconnect_from_host()
			tcp.connect_to_host(HOST, PORT)


# ---------- controls ----------

func controls(dt: float) -> Dictionary:
	var key := func(a: Key, b: Key) -> float: return float(Input.is_key_pressed(a) or Input.is_key_pressed(b))
	# pedals: triggers are analog; keys ramp up over a quarter second so a tap is a small input
	thr = move_toward(thr, key.call(KEY_W, KEY_UP), 4 * dt)
	brk = move_toward(brk, key.call(KEY_S, KEY_DOWN), 6 * dt)
	var t := maxf(thr, Input.get_joy_axis(0, JOY_AXIS_TRIGGER_RIGHT))
	var b := maxf(brk, Input.get_joy_axis(0, JOY_AXIS_TRIGGER_LEFT))
	# steering: stick with a dead zone and a soft centre, or keys that wind the wheel on and snap back faster
	var stick := Input.get_joy_axis(0, JOY_AXIS_LEFT_X)
	stick = 0.0 if absf(stick) < 0.08 else signf(stick) * pow(absf(stick), 2.0)
	var keys: float = key.call(KEY_A, KEY_LEFT) - key.call(KEY_D, KEY_RIGHT)
	if keys != 0 or stick == 0:
		steer = move_toward(steer, keys, (1.2 * sens if keys != 0 and signf(keys) == signf(steer) else 4.0) * dt)
	else:
		steer = move_toward(steer, -stick, 5 * sens * dt)
	# full lock only at low speed, like the F1 games: at 300 km/h full stick is about 5% of lock
	var v: float = cur.get("kmh", 0.0) / 3.6
	var lock := minf(1.0, 340.0 * minf(sens, 1.0) / maxf(v * v, 1.0))
	if b > 0.05:
		drs = false   # braking closes it, as in the real car
	var ot := Input.is_joy_button_pressed(0, JOY_BUTTON_B) or Input.is_key_pressed(KEY_SPACE)
	return {"steer": steer * lock, "x": steer, "tb": t - b, "drs": int(drs), "ot": int(ot)}


func _unhandled_input(e: InputEvent) -> void:
	if not e.is_pressed() or e.is_echo():
		return
	var k: int = e.keycode if e is InputEventKey else -1
	var j: int = e.button_index if e is InputEventJoypadButton else -1
	if k == KEY_F or j == JOY_BUTTON_Y:
		drs = not drs
	elif k == KEY_G or j == JOY_BUTTON_X:
		guide.visible = not guide.visible
	elif k == KEY_BRACKETLEFT or k == KEY_BRACKETRIGHT:
		sens = clampf(sens * (0.8 if k == KEY_BRACKETLEFT else 1.25), 0.3, 3.0)
		hud.track.text = "steering %d%%" % (sens * 100)
	elif k == KEY_C or j == JOY_BUTTON_RIGHT_SHOULDER:
		cam_i = (cam_i + 1) % CAMS.size()
	elif k == KEY_BACKSPACE or j == JOY_BUTTON_LEFT_SHOULDER:
		want_rescue = true
	elif k == KEY_R or j == JOY_BUTTON_BACK:
		want_reset = true
		prev = {}
		cur = {}
	elif k == KEY_P or k == KEY_ESCAPE or j == JOY_BUTTON_START:
		paused = not paused


# ---------- drawing ----------

func _process(dt: float) -> void:
	hud.msg.text = ""
	if not connected:
		hud.msg.text = "Waiting for the sim.\nRun:  python server.py monza"
		return
	if cur.is_empty():
		return
	var f := 1.0 if paused else Engine.get_physics_interpolation_fraction()
	while cars.size() < cur.cars.size():
		cars.append(build_car(CAR_COLORS[cars.size() % CAR_COLORS.size()]))
		var dot := ColorRect.new()
		dot.size = Vector2(8, 8)
		dot.color = CAR_COLORS[map_dots.size() % CAR_COLORS.size()]
		map_line.add_child(dot)
		map_dots.append(dot)
	for i in cur.cars.size():
		var a: Array = prev.cars[i] if i < prev.cars.size() else cur.cars[i]
		var b: Array = cur.cars[i]
		var xy := Vector2(a[0], a[1]).lerp(Vector2(b[0], b[1]), f)
		cars[i].position = Vector3(xy.x, 0, -xy.y)
		cars[i].rotation.y = lerp_angle(a[2], b[2], f)
		for w in cars[i].get_meta("front"):
			w.rotation.y = b[3]
		map_dots[i].position = (xy - map_mid) * Vector2(map_scale, -map_scale) - Vector2(4, 4)

	# camera: follows the car's position exactly and swings behind its heading with a little lag
	var c: Array = CAMS[cam_i]
	var back := Input.is_joy_button_pressed(0, JOY_BUTTON_RIGHT_STICK) or Input.is_key_pressed(KEY_B)
	rig.position = cars[0].position
	rig.rotation.y = lerp_angle(rig.rotation.y, cars[0].rotation.y, minf(1.0, c[2] * dt))
	cam.position = c[0] * (Vector3(-0.3, 1, 1) if back else Vector3.ONE)
	cam.rotation = Vector3(c[1], PI / 2 if back else -PI / 2, 0)
	cam.fov = 65 + cur.kmh * 0.05   # widens with speed

	hud.speed.text = "%d" % cur.kmh
	hud.gear.text = "%d" % cur.gear
	hud.lap.text = "LAP %d    %s" % [cur.lap, clock(cur.t)]
	hud.times.text = "last  %s\nbest  %s" % [clock(cur.last), clock(cur.best)]
	if cur.cars.size() > 1 and cur.gap > 0 and cur.gap < 30:
		hud.times.text += "\ngap   +%.2f" % cur.gap
	if cur.dmg > 0:
		hud.times.text += "\ndamage %d%%" % (cur.dmg * 100)
	hud.drs.modulate = Color.GREEN if cur.drs else (Color.WHITE if drs else Color(1, 1, 1, 0.25))
	hud.ot.modulate = Color.MAGENTA if cur.ot else Color(1, 1, 1, 0.25)
	hud.thr.size.x = 200 * cur.thr
	hud.brk.size.x = 200 * cur.brk
	hud.batt.size.x = 200 * cur.batt
	# guide cue: too fast for what is coming = brake, well under it and off the pedal = throttle
	var cue := 0 if not guide.visible else (2 if cur.kmh > cur.target + 8 else (1 if cur.kmh < cur.target - 8 and cur.thr < 0.9 else 0))
	hud.cue.text = ["", "THROTTLE", "BRAKE"][cue]
	hud.cue.modulate = [Color.WHITE, Color(0.2, 0.9, 0.3), Color(1, 0.2, 0.2)][cue]
	if cur.done:
		hud.msg.text = "CRASHED\npress R / Back to restart"
	elif paused:
		hud.msg.text = "PAUSED"
	if cur.off or cur.dmg > last_dmg:   # rumble in the gravel and on a hit
		Input.start_joy_vibration(0, 0.4, 0.8 if cur.dmg > last_dmg else 0.2, 0.15)
	last_dmg = cur.dmg


func clock(t: float) -> String:
	return "%d:%06.3f" % [int(t / 60), fmod(t, 60)] if t > 0 else "-:--.---"


## car.glb from Blender if game/car.glb exists (nose along +x after MODEL_YAW, front wheels named
## wheel_fl / wheel_fr so they steer); otherwise a car made of boxes.
func build_car(col: Color) -> Node3D:
	var car := Node3D.new()
	var front: Array[Node3D] = []
	if ResourceLoader.exists("res://car.glb"):
		var model: Node3D = load("res://car.glb").instantiate()
		model.rotation.y = MODEL_YAW
		car.add_child(model)
		for name in ["wheel_fl", "wheel_fr"]:
			var w := model.find_child(name)
			if w:
				front.append(w)
	else:
		var dark := Color(0.08, 0.08, 0.08)
		# [size, position, colour]: floor, body, nose, engine cover, front wing, rear wing, halo
		for p in [[Vector3(3.4, 0.12, 1.5), Vector3(0, 0.15, 0), dark], [Vector3(2.6, 0.45, 0.8), Vector3(-0.3, 0.42, 0), col],
				[Vector3(1.9, 0.25, 0.4), Vector3(1.9, 0.32, 0), col], [Vector3(1.4, 0.35, 0.4), Vector3(-1.0, 0.8, 0), col],
				[Vector3(0.5, 0.08, 1.9), Vector3(2.65, 0.15, 0), dark], [Vector3(0.35, 0.3, 1.1), Vector3(-2.5, 0.85, 0), dark],
				[Vector3(0.9, 0.06, 0.5), Vector3(0.3, 0.78, 0), dark]]:
			var m := MeshInstance3D.new()
			m.mesh = BoxMesh.new()
			m.mesh.size = p[0]
			m.mesh.material = flat(p[2])
			m.position = p[1]
			car.add_child(m)
		for x in [1.75, -1.75]:
			for z in [-0.8, 0.8]:
				var w := MeshInstance3D.new()
				w.mesh = CylinderMesh.new()
				w.mesh.top_radius = 0.36
				w.mesh.bottom_radius = 0.36
				w.mesh.height = 0.38
				w.mesh.material = flat(Color(0.05, 0.05, 0.05))
				w.position = Vector3(x, 0.36, z)
				w.rotation.x = PI / 2   # lay the cylinder on its side; rotation.y then steers it
				car.add_child(w)
				if x > 0:
					front.append(w)
	car.set_meta("front", front)
	add_child(car)
	return car


## One mesh of coloured strips along the lap: gravel, kerb, white line, asphalt, and a barrier each side.
func build_track(t: Dictionary) -> void:
	if track:
		track.queue_free()
	var n: int = t.x.size()
	var wall: float = t.wall
	var asphalt := Color(0.2, 0.2, 0.22)
	var gravel := Color(0.62, 0.55, 0.42)
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var at := func(i: int, off: float, h: float) -> Vector3:   # point `off` metres left of the centreline
		var c := Vector2(t.x[i % n], t.y[i % n])
		var tan := (Vector2(t.x[(i + 1) % n], t.y[(i + 1) % n]) - Vector2(t.x[(i - 1 + n) % n], t.y[(i - 1 + n) % n])).normalized()
		var p := c + Vector2(-tan.y, tan.x) * off
		return Vector3(p.x, h, -p.y)
	var pts := PackedVector2Array()
	for i in n:
		pts.append(Vector2(t.x[i], t.y[i]))
		var kerb: Color = (Color(0.85, 0.1, 0.1) if (i / 2) % 2 == 0 else Color.WHITE) if t.kerb[i] else gravel
		var l0: float = t.wl[i]
		var l1: float = t.wl[(i + 1) % n]
		var r0: float = -t.wr[i]
		var r1: float = -t.wr[(i + 1) % n]
		# [offset at i, offset at i+1, height] for each edge, right to left; a colour for each strip between two edges
		var edges := [[r0 - wall, r1 - wall, 1.0], [r0 - wall, r1 - wall, 0.0], [r0 - 1, r1 - 1, 0.0], [r0, r1, 0.0],
				[r0 + 0.15, r1 + 0.15, 0.0], [l0 - 0.15, l1 - 0.15, 0.0], [l0, l1, 0.0], [l0 + 1, l1 + 1, 0.0],
				[l0 + wall, l1 + wall, 0.0], [l0 + wall, l1 + wall, 1.0]]
		var cols := [Color(0.5, 0.5, 0.55), gravel, kerb, Color.WHITE, Color.WHITE if i == 0 else asphalt, Color.WHITE, kerb,
				gravel, Color(0.5, 0.5, 0.55)]
		for k in cols.size():
			var a: Vector3 = at.call(i, edges[k][0], edges[k][2])
			var b: Vector3 = at.call(i + 1, edges[k][1], edges[k][2])
			var c: Vector3 = at.call(i + 1, edges[k + 1][1], edges[k + 1][2])
			var d: Vector3 = at.call(i, edges[k + 1][0], edges[k + 1][2])
			st.set_color(cols[k])
			st.set_normal(Vector3.UP)
			for v in [a, c, b, a, d, c]:   # clockwise seen from above = facing up
				st.add_vertex(v)
	# guide ribbon on the racing line, just above the road
	var gt := SurfaceTool.new()
	gt.begin(Mesh.PRIMITIVE_TRIANGLES)
	var pace := [Color(0.1, 0.9, 0.25, 0.7), Color(1.0, 0.85, 0.1, 0.7), Color(1.0, 0.15, 0.1, 0.8)]
	for i in n:
		var l0: float = t.line[i]
		var l1: float = t.line[(i + 1) % n]
		gt.set_color(pace[int(t.pace[i])])
		for v in [at.call(i, l0 - 0.3, 0.03), at.call(i + 1, l1 + 0.3, 0.03), at.call(i + 1, l1 - 0.3, 0.03),
				at.call(i, l0 - 0.3, 0.03), at.call(i, l0 + 0.3, 0.03), at.call(i + 1, l1 + 0.3, 0.03)]:
			gt.add_vertex(v)
	var gm := StandardMaterial3D.new()
	gm.vertex_color_use_as_albedo = true
	gm.vertex_color_is_srgb = true
	gm.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	gm.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	gm.cull_mode = BaseMaterial3D.CULL_DISABLED
	gt.set_material(gm)
	guide.mesh = gt.commit()

	var mat := StandardMaterial3D.new()
	mat.vertex_color_use_as_albedo = true
	mat.vertex_color_is_srgb = true
	mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	st.set_material(mat)
	track = MeshInstance3D.new()
	track.mesh = st.commit()
	add_child(track)

	# minimap: the centreline squeezed into a 200 px box, top right
	var lo := pts[0]
	var hi := pts[0]
	for p in pts:
		lo = lo.min(p)
		hi = hi.max(p)
	map_mid = (lo + hi) / 2
	map_scale = 200.0 / maxf(hi.x - lo.x, hi.y - lo.y)
	for i in pts.size():
		pts[i] = (pts[i] - map_mid) * Vector2(map_scale, -map_scale)
	map_line.points = pts
	hud.track.text = str(t.track).to_upper() + ("   beginner assists" if t.assist else "   pro")


func build_hud() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	var label := func(pos: Vector2, size: int, text: String) -> Label:
		var l := Label.new()
		l.position = pos
		l.text = text
		l.add_theme_font_size_override("font_size", size)
		l.add_theme_color_override("font_outline_color", Color.BLACK)
		l.add_theme_constant_override("outline_size", 6)
		layer.add_child(l)
		return l
	var bar := func(pos: Vector2, col: Color) -> ColorRect:
		var back := ColorRect.new()
		back.position = pos
		back.size = Vector2(200, 12)
		back.color = Color(0, 0, 0, 0.5)
		layer.add_child(back)
		var r := ColorRect.new()
		r.position = pos
		r.size = Vector2(0, 12)
		r.color = col
		layer.add_child(r)
		return r
	hud.track = label.call(Vector2(30, 20), 22, "")
	hud.lap = label.call(Vector2(30, 50), 34, "")
	hud.times = label.call(Vector2(30, 100), 22, "")
	hud.speed = label.call(Vector2(1330, 700), 90, "0")
	label.call(Vector2(1335, 810), 20, "km/h")
	hud.gear = label.call(Vector2(1510, 720), 64, "1")
	hud.drs = label.call(Vector2(1100, 760), 30, "DRS")
	hud.ot = label.call(Vector2(1180, 760), 30, "OT")
	hud.thr = bar.call(Vector2(1100, 810), Color(0.2, 0.85, 0.3))
	hud.brk = bar.call(Vector2(1100, 828), Color(0.9, 0.2, 0.2))
	hud.batt = bar.call(Vector2(1100, 846), Color(1.0, 0.8, 0.1))
	hud.cue = label.call(Vector2(0, 180), 44, "")
	hud.cue.size = Vector2(1600, 60)
	hud.cue.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	hud.msg = label.call(Vector2(0, 330), 48, "")
	hud.msg.size = Vector2(1600, 200)
	hud.msg.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	map_line.position = Vector2(1470, 130)
	map_line.width = 3
	layer.add_child(map_line)
