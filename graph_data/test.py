import json

RL_data = {i: {"data": [], "sell": [], "buy": [], "profit": 0} for i in range(10)}


print(RL_data)

for i in range(10):
    print(i)
    file = open(f"data{i}.json", "r")
    data = json.load(file)
    RL_data[i]["data"] = data["data"]
    RL_data[i]["sell"] = data["sell"]
    RL_data[i]["buy"] = data["buy"]
    RL_data[i]["profit"] = data["profit"]
    RL_data[i]["total_profit"] = data["total_profit"]
    file.close()


file = "data.json"
with open(file, "w") as f:
    json.dump(RL_data, f)
