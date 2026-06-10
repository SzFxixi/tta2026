import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import math
class WallFitter:
    def __init__(self):
        rospy.init_node('wall_fitter')
        self.angle_increment = None
    
    def get_lidar_data(self):
        """获取激光雷达数据"""
        try:
            data = rospy.wait_for_message("scan", LaserScan, timeout=1.0)
            self.angle_increment = data.angle_increment
            ranges = np.array(data.ranges)
            angles = data.angle_min + np.arange(len(ranges)) * data.angle_increment
            
            # 过滤无效值并标记
            valid_mask = (ranges > data.range_min) & (ranges < data.range_max) & ~np.isnan(ranges)
            processed_ranges = np.where(valid_mask, ranges, -1)
            
            return processed_ranges, angles
        except rospy.ROSException:
            rospy.logwarn("LaserScan message timeout")
            return None, None
    
    def detect_mutations(self, ranges, width_threshold=0.1):
        """检测细杆等突变区域"""
        n = len(ranges)
        mutation_mask = np.zeros(n, dtype=bool)
        
        if self.angle_increment is None:
            return mutation_mask
        
        for i in range(1, n-1):
            if ranges[i] == -1:
                continue
                
            left = ranges[i-1]
            right = ranges[i+1]
            
            if left == -1 or right == -1:
                continue
                
            # 计算角度差对应的弧长
            arc_len_left = ranges[i] * self.angle_increment
            arc_len_right = ranges[i] * self.angle_increment
            
            # 突变检测条件
            is_mutation = (
                abs(ranges[i] - left) > 0.5 and
                abs(ranges[i] - right) > 0.5 and
                abs(left - right) < 0.2 and
                min(arc_len_left, arc_len_right) < width_threshold
            )
            
            if is_mutation:
                # 向前后扩展突变区域
                start = i
                end = i
                # 向前搜索
                for j in range(i-1, max(0, i-10), -1):
                    if abs(ranges[j] - ranges[i]) < 0.1:
                        start = j
                    else:
                        break
                # 向后搜索
                for j in range(i+1, min(n, i+10)):
                    if abs(ranges[j] - ranges[i]) < 0.1:
                        end = j
                    else:
                        break
                mutation_mask[start:end+1] = True
        
        return mutation_mask
    
    def fit_walls(self, ranges, angles):
        """拟合矩形房间的四壁"""
        if ranges is None or angles is None:
            return {}
        
        # 1. 过滤突变区域
        mutation_mask = self.detect_mutations(ranges)
        valid_mask = (ranges != -1) & ~mutation_mask
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]
        
        # 2. 转换为笛卡尔坐标
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        points = np.column_stack((x, y))
        
        # 3. 定义四个墙面区域
        walls = {
            'front': (-np.radians(30), np.radians(30)),
            'back': (np.radians(150), np.radians(210)),
            'left': (np.radians(60), np.radians(120)),
            'right': (np.radians(-120), np.radians(-60))
        }
        
        results = {}
        
        for name, (min_angle, max_angle) in walls.items():
            # 处理角度范围
            if min_angle < -np.pi:
                min_angle += 2*np.pi
            if max_angle > np.pi:
                max_angle -= 2*np.pi
            
            # 选择角度范围内的点
            if min_angle < max_angle:
                mask = (valid_angles >= min_angle) & (valid_angles <= max_angle)
            else:
                mask = (valid_angles >= min_angle) | (valid_angles <= max_angle)
            
            wall_points = points[mask]
            
            if len(wall_points) < 10:
                results[name] = {'slope': None, 'intercept': None, 'direction': None}
                continue
            
            # 4. 使用PCA辅助的线性拟合
            slope, intercept, direction = self.fit_wall(wall_points)
            results[name] = {
                'slope': slope,
                'intercept': intercept,
                'direction': direction
            }
        
        return results
    
    def fit_wall(self, points):
        """使用PCA辅助的墙面拟合"""
        # PCA确定主方向
        pca = PCA(n_components=2)
        pca.fit(points)
        vx, vy = pca.components_[0]
        
        # 根据主方向选择拟合模型
        if abs(vx) > abs(vy):  # 水平墙面
            X = points[:, 0].reshape(-1, 1)
            y = points[:, 1]
            model = LinearRegression()
            model.fit(X, y)
            slope = model.coef_[0]
            intercept = model.intercept_
            direction = "horizontal"
        else:  # 垂直墙面
            X = points[:, 1].reshape(-1, 1)
            y = points[:, 0]
            model = LinearRegression()
            model.fit(X, y)
            slope = model.coef_[0]
            intercept = model.intercept_
            direction = "vertical"
        
        return slope, intercept, direction
    
    def run(self):
        """主循环"""
        while not rospy.is_shutdown():
            ranges, angles = self.get_lidar_data()
            if ranges is None:
                rospy.sleep(0.5)
                continue
                
            wall_results = self.fit_walls(ranges, angles)
            k1 = 0
            k2 = 0
            # 打印结果
            for wall, params in wall_results.items():
                slope = params['slope']
                direction = params['direction']
                
                if slope is None:
                    rospy.loginfo(f"{wall}: 无法拟合")
                else:
                    # 对于垂直墙面，计算法线斜率
                    if direction == "vertical":
                        # 法线斜率 = -1 / 原斜率
                        normal_slope = -1 / slope if abs(slope) > 1e-5 else float('inf')
                        k2 += (1/normal_slope)/2
                        # rospy.loginfo(f"{wall} (垂直墙面): 斜率 = {1/normal_slope:.6f}")
                    else:
                        k1 += slope/2
                        # rospy.loginfo(f"{wall} (水平墙面): 斜率 = {slope:.6f}")
            alpha = (math.atan(k1) + math.atan(k2))/2*180/3.14159265358979 # 机器人转角
            print(f"转角：{alpha}")

            
            rospy.sleep(0.5)

if __name__ == "__main__":
    fitter = WallFitter()
    fitter.run()