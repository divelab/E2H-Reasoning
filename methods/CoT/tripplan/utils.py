
from typing import Dict
from tqdm import tqdm
import json
import re
from collections import defaultdict

def retrieve_answer_from_dataset(example):
  return [(city, int(distance)) for city, distance in zip(example['cities'].split('**'), example['durations'].split('**'))]
  

def parse_response(outputs):
    response = outputs[0]
    pattern_visit = r'\d+-\d+'
    pattern_flight = r'.*Day (\d+).*from (\w+) to (\w+)'
    pattern_days = r'European cities for (\d+) days'

    days, flights, flight_days = [], [], []
    total_days = None
    for piece in response.split('\n'):
        days_match = re.findall(pattern_days, piece)
        if days_match:
            total_days = int(days_match[0])

        visit_match = re.findall(pattern_visit, piece)
        if visit_match:
            days.append(visit_match[0])
            end_day = int(visit_match[0].split('-')[1])
            # Reach the end of the plan, stop to avoid parsing alternative plans.
            if end_day == total_days:
              break
        flight_match = re.findall(pattern_flight, piece)
        if flight_match:
            flights.append(flight_match[0])

    visit_cities, parsed_plan = [], []
    for flight_day, begin_city, end_city in flights:
        flight_days.append(int(flight_day))
        if not visit_cities:
            visit_cities.append(begin_city)
            visit_cities.append(end_city)
        else:
            visit_cities.append(end_city)

    if not days or not flights or not visit_cities:
        return []
    last_day = int(days[-1].split('-')[1])
    flight_days = [1] + flight_days + [last_day]
    for i, visit_city in enumerate(visit_cities):
        city_stay = flight_days[i + 1] - flight_days[i] + 1
        parsed_plan.append((visit_city, city_stay))
    return parsed_plan


def get_initial_config(cities,durations,prompt):
  
  def extract_time_windows(prompt, cities):
    # prompt = "You plan to visit 5 European cities for 22 days in total. You only take direct flights to commute between cities. You would like to visit Stuttgart for 6 days. During day 5 and day 10, you have to attend a conference in Stuttgart. You plan to stay in Riga for 2 days. You want to spend 6 days in Venice. From day 10 to day 15, there is a annual show you want to attend in Venice. You want to spend 7 days in Nice. You want to spend 5 days in Edinburgh.\n\nHere are the cities that have direct flights:\nVenice and Nice, Edinburgh and Stuttgart, Nice and Riga, Stuttgart and Venice, Edinburgh and Riga, Edinburgh and Venice, Edinburgh and Nice.\n\nFind a trip plan of visiting the cities for 22 days by taking direct flights to commute between them."
    pattern = r'Day\s*(\d+)\s*(and|to)\s*Day\s*(\d+)'
    time_windows = {}
    for sentence in prompt.lower().split('.'):
      pattern_match = re.search(pattern,sentence, re.IGNORECASE)
      if pattern_match:
        for city in cities:
          words = sentence.lower().split()
          if city.lower() in words:
            time_windows[city] = (int(pattern_match.group(1)), int(pattern_match.group(3)))     
    return time_windows
  
  def find_days_to_spend(cities, duration):
    days_to_spend = dict(zip(cities.split('**'), duration.split('**')))
    days_to_spend = {k: int(v) for k, v in days_to_spend.items()}
    return days_to_spend
  
  def find_total_days(prompt):
    pattern = r'\d+\s*days?\s*in\s*total'
    matches = re.findall(pattern, prompt, re.IGNORECASE)
    assert len(matches) == 1
    days = []
    for match in matches:
        digits = re.search(r'\d+', match).group()  # Extract the digits from the match
        days.append(int(digits))
    assert len(days) == 1
    return days[0]
  
  def get_flight_connections(prompt, cities):
    cities_pattern = '|'.join(cities)
    pattern = rf'({cities_pattern})\s*(?:and|to)\s*({cities_pattern})'
    
    
    adjacency_list = defaultdict(list)
    for sentence in prompt.split('.'):
      if sentence.lower().strip().startswith('here are the cities that have direct flights:'):
        matches = re.findall(pattern, sentence, re.IGNORECASE)
        for city1, city2 in matches:
          adjacency_list[city1].append(city2)
          # adjacency_list[city2].append(city1)
        
    adjacency_list = dict(adjacency_list)
    # assert len(adjacency_list) == len(cities), "Cities should match"
        
    return adjacency_list
  
  days_to_spend = find_days_to_spend(cities,durations)
  total_days = find_total_days(prompt)
  # print(sum(days_to_spend.values()) - len(days_to_spend.keys()) - 1, total_days) 
  assert total_days == sum(days_to_spend.values()) +1 - len(days_to_spend.keys()), "Total days don't match" # For N cities, N-1 flights
  time_windows = extract_time_windows(prompt, cities = days_to_spend.keys())
  flights = get_flight_connections(prompt, cities = days_to_spend.keys())
  initial_config = {
      "days_to_spend": days_to_spend,
      "total_days": total_days,
      "city_windows": time_windows,
      "flights": flights
  }
  return initial_config



    




