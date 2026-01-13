import heapq
from typing import Dict
from eval_helper import compute_score
import json


# Define a class to represent a travel state and include the dynamic heuristic
class TravelStateWithHeuristic:
    def __init__(
        self,
        current_city,
        days_spent,
        cities_visited,
        total_days,
        path,
        total_days_limit,
        days_to_spend,
        city_windows,
        visit_log={},
    ):
        self.current_city = current_city  # The city you are currently in
        self.days_spent = days_spent  # Dictionary tracking days spent in each city
        self.cities_visited = cities_visited  # Set of visited cities
        self.total_days = total_days  # Total days spent so far
        self.path = path  # Path taken so far
        self.total_days_limit = total_days_limit  # Total days limit
        self.days_to_spend = days_to_spend  # List of days to spend in each city
        self.city_windows = city_windows  # Dictionary of city windows
        self.visit_log = visit_log

        # g(n): Total days spent so far (cost)
        self.g = total_days
        self.total_cost = self.g
        # h(n): Heuristic (estimated days remaining or cities left to visit)
        self.h = self.human_heuristic()  # heuristic()
        self.total_cost += self.h

    def __lt__(self, other):
        return self.total_cost < other.total_cost

    def heuristic(self):
        # Heuristic: Remaining days and cities left to visit
        remaining_days = self.total_days_limit - self.total_days
        cities_left = len(self.days_to_spend) - len(self.cities_visited)

        # Penalty for arriving too early or too late to cities with window constraints
        penalty = 0
        for city, window in self.city_windows.items():
            if city not in self.cities_visited:
                # Early penalty if planning to visit the city before its window opens
                if self.total_days < window[0]:
                    penalty += abs(
                        window[0] - self.total_days
                    )  # The farther from the window, the bigger the penalty
                # Late penalty if visiting the city after its window closes
                elif self.total_days > window[1]:
                    penalty += 10  # High penalty for arriving after the window closes
        # print(self.current_city, self.visit_log)
        return remaining_days + cities_left + penalty

    def human_heuristic(self):
        # Heuristic: Remaining days and cities left to visit
        remaining_days = self.total_days_limit - self.total_days
        cities_left = len(self.days_to_spend) - len(self.cities_visited)

        # Penalty for arriving too early or too late to cities with window constraints
        penalty = 0
        for city, window in self.city_windows.items():
            if city not in self.cities_visited:
                if self.total_days > window[0]:
                    penalty += abs(self.total_days - window[0])
            else:
                delta = self.city_windows[city][1] - self.city_windows[city][0]
                if not (
                    self.visit_log[city][0] == self.city_windows[city][0]
                    and self.city_windows[city][1] == self.visit_log[city][0] + delta
                ):
                    penalty += (
                        abs(
                            self.city_windows[city][0]
                            - self.visit_log[city][0]
                            + self.city_windows[city][1]
                            - self.visit_log[city][0]
                            - delta
                        )
                        + self.total_days_limit
                    )
        heauristic = remaining_days + cities_left + penalty
        return heauristic


# Function to check if the travel plan meets the goal
def is_goal(state):
    return state.total_days == state.total_days_limit and len(
        state.cities_visited
    ) == len(state.days_to_spend)  # and state.h == 0


# Generic A* travel planning function that uses the new heuristic and does not explicitly block paths
def a_star_travel_with_dynamic_window_heuristic(
    initial_city, flights, days_to_spend, total_days_limit, city_windows=None
):
    if city_windows is None:
        city_windows = {}

    # Priority queue for A* (min-heap)
    pq = []
    visited = set()
    path_count = 0  # Counter for how many paths were visited
    total_states_visited = 0  # Counter for how many states were explored

    # Initial state using the modified heuristic class
    initial_state = TravelStateWithHeuristic(
        initial_city,
        {initial_city: 1},
        {initial_city},
        1,
        [initial_city],
        total_days_limit=total_days_limit,
        days_to_spend=days_to_spend,
        city_windows=city_windows,
        visit_log={initial_city: [1, 1]},
    )
    heapq.heappush(pq, initial_state)
    # A* Search
    while pq:
        current_state = heapq.heappop(pq)
        path_count += 1  # Increment the counter for each path visited
        total_states_visited += 1  # Increment the total number of states explored

        # Goal check
        if is_goal(current_state):
            return (
                current_state.path,
                current_state.total_days,
                path_count,
                total_states_visited,
                current_state.visit_log,
            )

        # Avoid re-visiting states
        state_key = (
            current_state.current_city,
            tuple(sorted(current_state.cities_visited)),
            current_state.total_days,
        )
        # print(state_key)
        if state_key in visited:
            continue
        visited.add(state_key)

        # Stay in the current city (spend one more day)
        if (
            current_state.days_spent[current_state.current_city]
            < days_to_spend[current_state.current_city]
        ):
            new_days_spent = current_state.days_spent.copy()
            new_days_spent[current_state.current_city] += 1
            visit_log = current_state.visit_log.copy()
            visit_log[current_state.current_city][1] += 1
            new_state = TravelStateWithHeuristic(
                current_state.current_city,
                new_days_spent,
                current_state.cities_visited.copy(),
                current_state.total_days + 1,
                current_state.path
                + [
                    f"Stay in {current_state.current_city} (Day {current_state.total_days + 1})"
                ],
                total_days_limit=total_days_limit,
                days_to_spend=days_to_spend,
                city_windows=city_windows,
                visit_log=visit_log,
            )
            heapq.heappush(pq, new_state)

        # Fly to a new city
        if (
            current_state.days_spent[current_state.current_city]
            == days_to_spend[current_state.current_city]
        ):
            if current_state.current_city in flights:
                for flight in flights[current_state.current_city]:
                    if flight not in current_state.cities_visited:
                        new_cities_visited = current_state.cities_visited.copy()
                        new_cities_visited.add(flight)
                        new_days_spent = current_state.days_spent.copy()

                        # Split the flight day between both cities
                        new_days_spent[current_state.current_city] += (
                            1  # Count the day in the departure city
                        )
                        new_days_spent[flight] = (
                            1  # Start counting the day in the arrival city
                        )

                        visit_log = current_state.visit_log.copy()
                        visit_log[flight] = [
                            current_state.total_days,
                            current_state.total_days,
                        ]
                        # if current_state.current_city == 'Seville' and 'Seville' in visit_log and visit_log['Seville'][1] == 9:
                        #     print(flight, visit_log, current_state.current_city, current_state.h)
                        new_state = TravelStateWithHeuristic(
                            flight,
                            new_days_spent,
                            new_cities_visited,
                            current_state.total_days,  # Do not increment the day since it's counted for both cities
                            current_state.path
                            + [f"Fly to {flight} (Day {current_state.total_days})"],
                            total_days_limit=total_days_limit,
                            days_to_spend=days_to_spend,
                            city_windows=city_windows,
                            visit_log=visit_log,
                        )
                        heapq.heappush(pq, new_state)

    return None, None, path_count, total_states_visited, None  # No solution found


# Update find_best_start_city function to use the new A* function
def find_best_start_city_dynamic(
    flights, days_to_spend, total_days_limit, city_windows=None
):
    if city_windows is None:
        city_windows = {}

    for start_city in days_to_spend.keys():  # Loop through all potential start cities
        path, total_days, path_count, total_states_visited, visit_log = (
            a_star_travel_with_dynamic_window_heuristic(
                start_city, flights, days_to_spend, total_days_limit, city_windows
            )
        )
        if path:
            print(
                f"Valid plan found starting from {start_city} and states visited: {total_states_visited}"
            )
            # print(f"Total paths visited: {path_count}")
            # print(f"Total states visited: {total_states_visited}")
            return start_city, path, total_days, visit_log
        quit()
    return None, None, None, None  # No valid plan found


def format_plan(visit_log: Dict[str, list]):
    formatted_plan = "Here is the trip plan for visiting the European cities:\n\n"
    for i, (city, value) in enumerate(visit_log.items()):
        formatted_plan = formatted_plan.replace("<new_city>", city)
        # print(formatted_plan, value[0], value[1])
        if i == 0:
            formatted_plan += f"**Day {value[0]}-{value[1]}:** Arriving in {city} and visit {city} for {value[1] - value[0] + 1} days.\n"
        else:
            formatted_plan += f"**Day {value[0]}-{value[1]}:** Visit {city} for {value[1] - value[0] + 1} days.\n"

        if i != len(visit_log) - 1:
            formatted_plan += f"**Day {value[1]}:** Fly from {city} to <new_city>.\n"
    return formatted_plan


def get_trip_plan(total_days, days_to_spend, flights, city_windows):
    start_city, path, total_days, visit_log = find_best_start_city_dynamic(
        flights, days_to_spend, total_days, city_windows
    )

    if path:
        # print(f"Total Days: {total_days}")
        formatted_plan = format_plan(visit_log)
    else:
        formatted_plan = "No Trip Plan Found."

    return formatted_plan


if __name__ == "__main__":
    with open("examples/CoT/tripplan/data/test_TripPlan-cities-3.json", "r") as f:
        data = json.load(f)

    cities, durations, responses = [], [], []
    sample_count = 0

    for i, (key, item) in enumerate(data.items()):
        # if len(item['cities'].split('**')) < num_steps:
        #   continue
        # if len(item['cities'].split('**')) > num_steps:
        #   break
        print(f"Starting : {key}")
        cities.append(item["cities"])
        durations.append(item["durations"])
        initial_config = item["initial_config"]
        cot = item["prompt_5shot"]
        print(cot.split("TASK:"))
        if initial_config:
            plan = get_trip_plan(**initial_config)
        else:
            plan = ""
        responses.append(plan)
        sample_count += 1
        item["initial_config"] = initial_config
        item["a_star_plan"] = plan

    hard_acc = compute_score(cities, durations, responses)
    print(hard_acc)
