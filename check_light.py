import pickle
from pathlib import Path
from pprint import pprint

path = Path("data/light_dialogue/light_data.pkl")

with path.open("rb") as f:
    episodes = pickle.load(f)

print(type(episodes))
print("에피소드 수:", len(episodes))

print("\n첫 번째 episode 타입:")
print(type(episodes[1]))

print("\n첫 번째 episode의 key:")
print(episodes[1].keys())

print("\n첫 번째 episode:")
pprint(episodes[1])