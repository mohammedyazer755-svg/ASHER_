from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ModelValidationReport:
    is_valid: bool
    is_production_rigged: bool
    mesh_count: int
    vertex_count: int
    bone_count: int
    has_animations: bool
    has_morph_targets: bool
    file_size_bytes: int
    error: str | None = None


def validate_companion_model(path: Path | str) -> ModelValidationReport:
    """Validate that a companion 3D model is a genuine glTF/GLB with real geometry and rigging.

    Rejects empty stubs, placeholder triangles (< 10 KB), and unrigged static files.
    """
    model_path = Path(path)
    if not model_path.is_file():
        return ModelValidationReport(
            is_valid=False,
            is_production_rigged=False,
            mesh_count=0,
            vertex_count=0,
            bone_count=0,
            has_animations=False,
            has_morph_targets=False,
            file_size_bytes=0,
            error=f"Model file does not exist: {model_path}",
        )

    size = model_path.stat().st_size
    # Production humanoid characters with geometry and skeleton are significantly larger than 10 KB
    if size < 10 * 1024:
        return ModelValidationReport(
            is_valid=False,
            is_production_rigged=False,
            mesh_count=0,
            vertex_count=0,
            bone_count=0,
            has_animations=False,
            has_morph_targets=False,
            file_size_bytes=size,
            error=f"Model file size ({size} bytes) is a placeholder stub (< 10 KB); real character geometry required",
        )

    try:
        data = model_path.read_bytes()
        if len(data) < 20:
            return ModelValidationReport(
                is_valid=False,
                is_production_rigged=False,
                mesh_count=0,
                vertex_count=0,
                bone_count=0,
                has_animations=False,
                has_morph_targets=False,
                file_size_bytes=size,
                error="File too small for valid GLB header",
            )

        magic, version, length = struct.unpack_from("<4sII", data, 0)
        if magic != b"glTF" or version != 2:
            return ModelValidationReport(
                is_valid=False,
                is_production_rigged=False,
                mesh_count=0,
                vertex_count=0,
                bone_count=0,
                has_animations=False,
                has_morph_targets=False,
                file_size_bytes=size,
                error=f"Invalid GLB header: magic={magic!r}, version={version}",
            )

        chunk0_len, chunk0_type = struct.unpack_from("<II", data, 12)
        if chunk0_type != 0x4E4F534A:  # "JSON"
            return ModelValidationReport(
                is_valid=False,
                is_production_rigged=False,
                mesh_count=0,
                vertex_count=0,
                bone_count=0,
                has_animations=False,
                has_morph_targets=False,
                file_size_bytes=size,
                error="First chunk is not JSON",
            )

        json_bytes = data[20 : 20 + chunk0_len]
        gltf: Mapping[str, Any] = json.loads(json_bytes.decode("utf-8", errors="replace"))

        meshes = gltf.get("meshes", [])
        mesh_count = len(meshes)
        if mesh_count == 0:
            return ModelValidationReport(
                is_valid=False,
                is_production_rigged=False,
                mesh_count=0,
                vertex_count=0,
                bone_count=0,
                has_animations=False,
                has_morph_targets=False,
                file_size_bytes=size,
                error="GLB contains no meshes",
            )

        # Count vertices across accessors
        accessors = gltf.get("accessors", [])
        vertex_count = 0
        has_morph_targets = False
        for mesh in meshes:
            for prim in mesh.get("primitives", []):
                pos_idx = prim.get("attributes", {}).get("POSITION")
                if pos_idx is not None and pos_idx < len(accessors):
                    vertex_count += accessors[pos_idx].get("count", 0)
                if "targets" in prim:
                    has_morph_targets = True

        # Check for skins and bones
        skins = gltf.get("skins", [])
        bone_count = 0
        for skin in skins:
            bone_count += len(skin.get("joints", []))

        # Check for animations
        animations = gltf.get("animations", [])
        has_animations = len(animations) > 0

        # A production character needs significant geometry (> 500 vertices) and skeletal bones
        is_production_rigged = (vertex_count >= 500 and bone_count >= 10)

        return ModelValidationReport(
            is_valid=True,
            is_production_rigged=is_production_rigged,
            mesh_count=mesh_count,
            vertex_count=vertex_count,
            bone_count=bone_count,
            has_animations=has_animations,
            has_morph_targets=has_morph_targets,
            file_size_bytes=size,
            error=None if is_production_rigged else "Model lacks sufficient vertices or skeletal bones for production character",
        )
    except Exception as exc:
        return ModelValidationReport(
            is_valid=False,
            is_production_rigged=False,
            mesh_count=0,
            vertex_count=0,
            bone_count=0,
            has_animations=False,
            has_morph_targets=False,
            file_size_bytes=size,
            error=f"GLB parsing error: {exc}",
        )
