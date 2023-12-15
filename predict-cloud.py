import requests

host = "predict-employee-turnover.onrender.com"
url = f"http://{host}/predict"

employee = {
    "education": "Masters",
    "joiningyear": 2017,
    "city": "Bangalore",
    "paymenttier": 3,
    "age": 34,
    "gender": "Male",
    "everbenched": "No",
    "experienceincurrentdomain": 0
}


response = requests.post(url, json=employee).json()
print(response)