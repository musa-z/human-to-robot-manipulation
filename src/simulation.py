import time
import mujoco
import mujoco.viewer
import numpy as np

MODEL_XML = """
<mujoco model="human_to_robot_manipulation">
    <option gravity="0 0 -9.81" timestep="0.002"/>

    <visual>
        <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3"/>
        <global azimuth="135" elevation="-25"/>
    </visual>

    <worldbody>
        <!-- Ground -->
        <geom name="ground" type="plane" pos="0 0 0"
              size="2 2 0.1" rgba="0.85 0.87 0.9 1"/>

        <!-- Table -->
        <body name="table" pos="0.45 0 0.35">
            <geom type="box" size="0.35 0.3 0.03"
                  rgba="0.45 0.28 0.18 1"/>
        </body>

        <!-- Robot base -->
        <body name="base" pos="0 0 0.05">
            <geom type="cylinder" size="0.08 0.05"
                  rgba="0.15 0.17 0.22 1"/>

            <!-- First link -->
            <body name="link1" pos="0 0 0.1">
                <joint name="joint1" type="hinge"
                       axis="0 0 1" range="-180 180"/>
                <geom type="capsule" fromto="0 0 0 0.15 0 0.2"
                      size="0.035" rgba="0.1 0.35 0.85 1"/>

                <!-- Second link -->
                <body name="link2" pos="0.15 0 0.2">
                    <joint name="joint2" type="hinge"
                           axis="0 1 0" range="-90 90"/>
                    <geom type="capsule" fromto="0 0 0 0.22 0 0"
                          size="0.03" rgba="0.1 0.65 0.4 1"/>

                    <!-- End effector -->
                    <body name="gripper" pos="0.22 0 0">
                        <geom name="gripper_visual" type="sphere"
                              size="0.045"
                              rgba="0.9 0.15 0.15 1"/>
                    </body>
                </body>
            </body>
        </body>

        <!-- Manipulation target -->
        <body name="block" pos="0.4 0 0.42">
            <freejoint/>
            <geom type="box" size="0.035 0.035 0.035"
                  mass="0.1" rgba="1 0.6 0.05 1"/>
        </body>
    </worldbody>

    <actuator>
        <position name="motor1" joint="joint1"
                  kp="30" ctrlrange="-3.14 3.14"/>
        <position name="motor2" joint="joint2"
                  kp="30" ctrlrange="-1.57 1.57"/>
    </actuator>
</mujoco>
"""


def main():
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)

    print("Robot simulation running.")
    print("The robot arm will move automatically.")
    print("Close the MuJoCo window to stop.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set a useful initial camera position.
        viewer.cam.lookat[:] = [0.25, 0, 0.35]
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20

        start = time.time()

        while viewer.is_running():
            elapsed = time.time() - start

            # Smoothly move the robot joints.
            data.ctrl[0] = 0.6 * np.sin(elapsed * 0.8)
            data.ctrl[1] = 0.4 * np.sin(elapsed * 0.6)

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)


if __name__ == "__main__":
    main()