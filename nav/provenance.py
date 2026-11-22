"""Content hashes of a scene's XML/include/asset closure, including training robot files."""
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET


def scene_files(scene, dynamics="training"):
    from .sim import scene_path
    from .policies import microduck_root
    root = Path(scene_path(str(scene))).resolve()
    pending = [(root, None, None)]
    # Training replaces the deployment robot include; hash both so either edit invalidates caches.
    if dynamics == "training":
        robot = microduck_root() / "upstream/rl_cb8561f6/src/mjlab_microduck/robot/microduck"
        pending.append((robot / "robot_walk_backlash.xml", robot / "assets", robot / "assets"))
    seen = set()
    while pending:
        path, meshdir, texturedir = pending.pop()
        path = path.resolve()
        if path in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"scene closure asset missing: {path}")
        seen.add(path)
        if path.suffix.lower() != ".xml":
            continue
        xml = ET.parse(path).getroot()
        compiler = xml.find("compiler")
        if compiler is not None:
            assetdir = compiler.get("assetdir", "")
            if compiler.get("meshdir") is not None or assetdir:
                meshdir = path.parent / compiler.get("meshdir", assetdir)
            if compiler.get("texturedir") is not None or assetdir:
                texturedir = path.parent / compiler.get("texturedir", assetdir)
        for element in xml.iter():
            file = element.get("file")
            if not file:
                continue
            if element.tag == "include":
                pending.append((path.parent / file, meshdir, texturedir))
            elif element.tag in ("mesh", "texture", "hfield", "skin"):
                base = meshdir if element.tag in ("mesh", "skin") else texturedir
                pending.append(((base or path.parent) / file, None, None))
    return sorted(seen)


def scene_sha256(scene, dynamics="training"):
    digest = hashlib.sha256()
    for path in scene_files(scene, dynamics):
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
