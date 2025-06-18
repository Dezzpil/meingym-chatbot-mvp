import requests
import json
from typing import Dict, List, Optional, Any, Union

from demo.structures import Result


class ApiClient:
    """HTTP client for interacting with the chatbot API."""
    
    def __init__(self, base_url: str = "http://localhost:3004"):
        self.base_url = base_url

    def create_training_from_llm(self, user_id: str, result: Result, muscles: List[str] = []):
        r = result.model_dump()
        return self.create_training(
            user_id=user_id,
            muscle_group=muscles,
            exercises=r["exercises"],
            comments=r["comments"]
        )

    def create_training(self, 
                         user_id: str, 
                         muscle_group: List[str],
                         exercises: List[str], 
                         comments: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new training program by sending a POST request to the API.
        
        Args:
            user_id: The ID of the user
            muscle_group: The muscle group for the training
            exercises: List of exercises for the training
            comments: Optional comments about the training
            
        Returns:
            Dict containing response data or error information
            
        Raises:
            requests.RequestException: If there's a network error
        """
        endpoint = f"{self.base_url}/api/chatbot/trainings"
        
        payload = {
            "userId": user_id,
            "muscleGroup": muscle_group,
            "exercises": exercises
        }
        
        if comments:
            payload["comments"] = comments
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(endpoint, headers=headers, data=json.dumps(payload))
            
            result = {
                "status_code": response.status_code,
                "success": False,
                "message": "",
                "data": None
            }
            
            # Handle different status codes
            if response.status_code == 200:
                result["success"] = True
                result["message"] = "Training created successfully"
                result["data"] = response.json()
                return result
            elif response.status_code == 400:
                result["message"] = "Bad request. Check your input data."
                return result
            elif response.status_code == 401:
                result["message"] = "Unauthorized. Authentication required."
                return result
            elif response.status_code == 403:
                result["message"] = "Forbidden. You don't have permission to access this resource."
                return result
            else:
                result["message"] = f"Unexpected status code: {response.status_code}"
                return result
                
        except requests.RequestException as e:
            return {
                "status_code": None,
                "success": False,
                "message": f"Request error: {str(e)}",
                "data": None
            }


# Example usage
if __name__ == "__main__":
    client = ApiClient()
    result = client.create_training(
        user_id="clux8gey10000wiqegntxp3y3",
        muscle_group="ноги",
        exercises=["Приседания со штангой", "Жим ногами"],
        comments="Комментарий к тренировке"
    )
    
    if result["success"]:
        print(f"Training created successfully. Path: {result['data'].get('path')}")
    else:
        print(f"Error: {result['message']}")
