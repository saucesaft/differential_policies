"""
the XMLs in assets/unitree_g1/ are mujoco_playground's feet-only scene; their
meshes are not vendored.
"""
import pathlib

import mujoco

from . import g1_29dof_dir, g1_29dof_xml

MENAGERIE_MODEL = 'unitree_g1'
# entry point whose asset set is read; 'g1' covers all 35 meshes the scene needs.
MENAGERIE_ENTRY = 'g1'
# model revision this env was verified against (mujoco_menagerie.get(...).oid).
MENAGERIE_OID = '57c00d310bfd8ae7d5676c64b959c86fbdd61d20'


def get_assets() -> dict:
    """{basename: bytes} for every file the scene needs"""
    try:
        import mujoco_menagerie
    except ImportError as e:
        raise ImportError(
            'The G1 env needs the mujoco-menagerie package for its meshes: '
            'pip install mujoco-menagerie==2026.9.0'
        ) from e

    robot = mujoco_menagerie.get(MENAGERIE_MODEL)
    if robot.oid != MENAGERIE_OID:
        # not fatal -- the meshes rarely change -- but worth knowing about if
        # the model's mass/inertia ever shifts under you.
        print(f'[g1_model] warning: mujoco-menagerie {MENAGERIE_MODEL} is at '
              f'oid {robot.oid}, env was verified against {MENAGERIE_OID}')

    assets = {pathlib.PurePath(k).name: v
              for k, v in robot.assets(MENAGERIE_ENTRY).items()}
    for f in pathlib.Path(g1_29dof_dir).glob('*.xml'):
        assets[f.name] = f.read_bytes()
    return assets


def build_g1_mj_model() -> mujoco.MjModel:
    """(nq=36, nv=35, nu=29)"""
    return mujoco.MjModel.from_xml_string(
        pathlib.Path(g1_29dof_xml).read_text(), assets=get_assets())
