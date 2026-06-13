import requests
import json
import time
from typing import Optional, Tuple, Dict
class CarController:
    
    def __init__(self, ip: str = "10.203.94.227", port: int = 5000):
        self.base_url = f"http://{ip}:{port}"
        self.current_task_id = 1
        self.session = requests.Session()
        

        self.task_history: Dict[int, Tuple[str, dict]] = {}
    
    def _retry_until_success(self, task_id: int) -> Tuple[bool, Optional[dict]]:

        attempts = 0
        while attempts < 10000:  
            if task_id in self.task_history:
                endpoint, payload = self.task_history[task_id]
                
                try:

                    response = self.session.post(
                        f"{self.base_url}/{endpoint}",
                        json=payload,
                        timeout=3000
                    )
                    

                    result = response.json()
                    

                    if result.get("isSuccess"):
                        return True, result
                except (requests.RequestException, json.JSONDecodeError):
                    pass
            time.sleep(0.1)
            attempts += 1
        return False, None

    def _send_command(self, endpoint: str, payload: dict) -> Tuple[bool, Optional[dict]]:


        
        new_task_id = self.current_task_id
        

        self.task_history[new_task_id] = (endpoint, payload.copy())
        payload["TaskId"] = new_task_id
        
        try:

            response = self.session.post(
                f"{self.base_url}/{endpoint}",
                json=payload,
                timeout=3000
            )
            

            result = response.json()
            
            

            if not result.get("isSuccess") and "expectedTaskId" in result:
                expected_id = result["expectedTaskId"]
                self.current_task_id = expected_id
                self._handle_missing_tasks()
                return self._send_command(endpoint, payload)
            self.current_task_id += 1

            return result.get("isSuccess", False), result
        
        except (requests.RequestException, json.JSONDecodeError):
            return self._retry_until_success(new_task_id)
    
    def _handle_missing_tasks(self):
        expected_id = self.current_task_id
        while True:
            if expected_id in self.task_history:
                endpoint, payload = self.task_history[expected_id]
                success, _ = self._send_command(endpoint, payload)
                if success:
                    expected_id += 1
            else:
                break

    def Circle(self,rad_z:float) -> bool:
        success, _ = self._send_command("Circle", {"rad_z": rad_z})
        return success
    def Move(self,location_x: float,location_y:float) -> bool:
        success, _ = self._send_command("Move", {"location_x": location_x, "location_y": location_y})
        return success
    def MoveOnlyX(self,location_x: float,location_y:float) -> bool:
        success, _ = self._send_command("MoveOnlyX", {"location_x": location_x, "location_y": location_y})
        return success
    def MoveOnlyY(self,location_x: float,location_y:float) -> bool:
        success, _ = self._send_command("MoveOnlyY", {"location_x": location_x, "location_y": location_y})
        return success
    
    def MoveLongDistance(self,location_x: float,location_y:float) -> bool:
        success, _ = self._send_command("MoveLongDistance", {"location_x": location_x, "location_y": location_y})
        return success
    
    def Shutdown(self):
        success, _ = self._send_command("ShutDown",{})
        self.session.close()

        self.task_history.clear()
        self.current_task_id = 1
        return success
    
    def Reset(self) -> bool:
        success, _ = self._send_command("Reset", {})
        self.task_history.clear()
        self.current_task_id = 1
        return success


if __name__ == "__main__":
    carController = CarController()
    
# Initialization 0.7 2
# the point between Initialization and position I 3.28 2
# position I 3.28 1.2
# the point between position II and position I 3.28 2.8
# position II 5.75 2.8
    carController.Move(0.7,2)

    carController.MoveOnlyX(3.1,2)
    carController.MoveOnlyY(3.1,1.2)
    carController.Move(3.28,1.2)

    carController.MoveOnlyX(4,1.2)
    carController.MoveOnlyY(4,2)
    carController.MoveOnlyX(5.45,2)
    carController.MoveOnlyY(5.45,2.8)
    carController.Move(5.75,2.8)

    carController.MoveOnlyX(6.2,2.8)
    carController.MoveOnlyY(6.2,2)
    carController.Move(7.84,2)

    carController.MoveLongDistance(0.7,2)


    carController.Reset()
