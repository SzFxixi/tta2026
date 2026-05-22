from typing import Any, Dict, List


class RescuePointManager:
    """救援点数据管理 — 记录每个救援点的等级识别结果，输出规则要求的 CSV 格式。"""

    def __init__(self, rescue_points_config: List[Dict[str, Any]]):
        self.points: Dict[str, Dict[str, Any]] = {}
        for pt in rescue_points_config:
            name = pt["name"]
            self.points[name] = {
                "grade": "unknown",
                "confidence": 0.0,
                "scanned": False,
                "image_path": "",
            }

    def set_result(self, point_name: str, grade: str, confidence: float = 0.0, image_path: str = "") -> None:
        if point_name not in self.points:
            raise KeyError(f"未知救援点: {point_name}")
        self.points[point_name].update({
            "grade": grade,
            "confidence": confidence,
            "scanned": True,
            "image_path": image_path,
        })

    def get_result(self, point_name: str) -> Dict[str, Any]:
        return self.points.get(point_name, {})

    def all_scanned(self) -> bool:
        return all(p["scanned"] for p in self.points.values())

    def to_csv_rows(self) -> List[List[str]]:
        """按规则格式输出: ['救援点', '救援等级'] → [['救援点1', '1级'], ...]"""
        rows = [["救援点", "救援等级"]]
        for name, info in self.points.items():
            level = f"{info['grade']}级" if info["grade"] != "unknown" else "unknown"
            rows.append([name, level])
        return rows

    def summary(self) -> Dict[str, Any]:
        return {name: {"grade": info["grade"], "scanned": info["scanned"]}
                for name, info in self.points.items()}
