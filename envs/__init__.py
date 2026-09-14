from pathlib import Path

slider_motor_xml = str(Path(Path(__file__).parent, Path("shac_test"), Path("slider_motor.xml")))
vslider_motor_xml = str(Path(Path(__file__).parent, Path("shac_test"), Path("slider_motor_vision.xml")))
slider_position_xml = str(Path(Path(__file__).parent, Path("shac_test"), Path("slider_position.xml")))
anymal_xml = str(Path(Path(__file__).parent, Path("assets/anybotics_anymal_c"), Path("anymal_c_torque.xml")))
g1_29dof_dir = str(Path(Path(__file__).parent, Path("assets/unitree_g1")))
g1_29dof_xml = str(Path(g1_29dof_dir, Path("scene_mjx_feetonly_flat_terrain.xml")))
