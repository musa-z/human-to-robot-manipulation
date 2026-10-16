import time
import mujoco
import mujoco.viewer

MODEL_PATH = "assets/franka_emika_panda/manipulation_scene.xml"


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    print("Panda manipulation simulation running.")
    print("Close the MuJoCo window to stop.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.3, 0, 0.35]
        viewer.cam.distance = 1.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20

        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = (
                model.opt.timestep - (time.time() - step_start)
            )

            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()