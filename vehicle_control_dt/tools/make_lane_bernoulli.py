import math
import numpy as np


def bernoulli_points(num_points=600, a=0.5):
    t = np.linspace(0.0, 2.0 * math.pi, num_points, endpoint=False)

    s = np.sin(t)
    c = np.cos(t)
    denom = 1.0 + s * s

    x = a * math.sqrt(2.0) * c / denom
    y = a * math.sqrt(2.0) * c * s / denom

    return np.stack([x, y], axis=1).astype(np.float32)


def make_road_obj(
    out_path="bernoulli_new.obj",
    num_points=600,
    a=0.5,
    lane_width=0.25,
    thickness=0.05,
):
    center = bernoulli_points(num_points=num_points, a=a)

    # tangent
    prev = np.roll(center, 1, axis=0)
    nxt = np.roll(center, -1, axis=0)
    tangent = nxt - prev

    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    tangent = tangent / norm

    # left normal
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)

    half_w = lane_width * 0.5

    left = center + normal * half_w
    right = center - normal * half_w

    verts = []

    # per point:
    # top left, top right, bottom left, bottom right
    for i in range(num_points):
        lx, ly = left[i]
        rx, ry = right[i]

        verts.append((lx, ly, 0.0))
        verts.append((rx, ry, 0.0))
        verts.append((lx, ly, -thickness))
        verts.append((rx, ry, -thickness))

    faces = []

    def vi(i, k):
        # OBJ index starts from 1
        return i * 4 + k + 1

    for i in range(num_points):
        j = (i + 1) % num_points

        tl0 = vi(i, 0)
        tr0 = vi(i, 1)
        bl0 = vi(i, 2)
        br0 = vi(i, 3)

        tl1 = vi(j, 0)
        tr1 = vi(j, 1)
        bl1 = vi(j, 2)
        br1 = vi(j, 3)

        # top
        faces.append((tl0, tr0, tr1))
        faces.append((tl0, tr1, tl1))

        # bottom
        faces.append((bl0, br1, br0))
        faces.append((bl0, bl1, br1))

        # left side
        faces.append((tl0, tl1, bl1))
        faces.append((tl0, bl1, bl0))

        # right side
        faces.append((tr0, br1, tr1))
        faces.append((tr0, br0, br1))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Bernoulli lemniscate road mesh\n")
        f.write(f"# num_points={num_points}, a={a}, lane_width={lane_width}, thickness={thickness}\n")

        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print(f"saved: {out_path}")
    print(f"vertices={len(verts)} faces={len(faces)}")


if __name__ == "__main__":
    make_road_obj(
        out_path="tools/bernoulli_a50_lane025_new.obj",
        num_points=600,
        a=0.5,
        lane_width=0.25,
        thickness=0.05,
    )