import pygame
import sys
from collections import deque
import heapq
import random
import math
import time
import numpy as np


import algorithms

SHIP_SPEED = 35
BULLET_SPEED = 35
BULLET_COOLDOWN = 500
BLACK = (0, 0, 0)

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

stepsUCS = 0
stepsDFS = 0
stepsBFS = 0
stepsIDS = 0
stepsAStar= 0
stepsAStar2= 0
stepsGreedy = 0
stepsGreedy2 = 0
stepsHill= 0
stepsAnnealing= 0
stepsGenetic = 0

timeUCS = 0
timeDFS = 0
timeBFS = 0
timeIDS = 0
timeAStar= 0
timeAStar2= 0
timeGreedy = 0
timeGreedy2 = 0
timeHill= 0
timeGenetic = 0

maxFrontierUCS = 0
maxFrontierDFS = 0
maxFrontierBFS = 0
maxFrontierIDS = 0
maxFrontierAStar= 0
maxFrontierAStar2= 0
maxFrontierGreedy = 0
maxFrontierGreedy2 = 0
maxFrontierHill= 0

#Game environment resposible for all game aspects
class GameEnvironment:
    def __init__(self, width=800, height=600):
        pygame.init()
        #attributes for the GUI
        self.SCREEN_WIDTH = width
        self.SCREEN_HEIGHT = height
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Chicken Invaders")
        self.font = pygame.font.SysFont("Arial", 24)
        self.last_bullet_time = 0
        self.spaceship_image = self.load_image("spaceship3.png", (60, 60))
        self.chicken_image = self.load_image("chicken.png", (40, 40))
        
        self.ship_x = (self.SCREEN_WIDTH - 60) // 2
        self.ship_y = self.SCREEN_HEIGHT - 60 - 10
        self.bullets = []
        self.initial_chickens = self.create_chickens()
        self.chickens = self.initial_chickens.copy()  
        self.last_shot_time = pygame.time.get_ticks()
        self.active_bullet = None

    def display_text(self, text, x, y, color=(255, 255, 255)):
        rendered_text = self.font.render(text, True, color)
        self.screen.blit(rendered_text, (x, y))

    def load_image(self, path, size):
 
        try:
            image = pygame.image.load(path)
            return pygame.transform.scale(image, size)
        except FileNotFoundError:
            print(f"Make sure '{path}' is in the same folder as this script!")
            sys.exit()

    def create_chickens(self): #creates the chiken grid

        chickens = []
        num = random.random()
        for i in range(5):  
            for j in range(5):
                num = random.random() 
                if (num > 0.55):
                    chickens.append(pygame.Rect(100 + j * (40 + 10), 50 + i * (40 + 10), 40, 40))
        return chickens

    def reset(self):

        self.ship_x = (self.SCREEN_WIDTH - 60) // 2
        self.ship_y = self.SCREEN_HEIGHT - 60 - 10
        self.bullets = []
        self.chickens = self.initial_chickens.copy()

        self.active_bullet = None
        self.last_bullet_time = 0
        self.last_shot_time = pygame.time.get_ticks()

    def draw_spaceship(self):

        self.screen.blit(self.spaceship_image, (self.ship_x, self.ship_y))

    def draw_chickens(self):

        for chicken in self.chickens:
            self.screen.blit(self.chicken_image, (chicken.x, chicken.y))

    def draw_bullet(self):

        if self.active_bullet:
            pygame.draw.rect(self.screen, (255, 0, 0), self.active_bullet)

    def shoot(self):    

        current_time = pygame.time.get_ticks()
        if current_time - self.last_shot_time > BULLET_COOLDOWN:
            bullet = pygame.Rect(self.ship_x + 30, self.ship_y, 5, 10)
            self.bullets.append(bullet)
            self.last_shot_time = current_time

class Node: #class node represnting each individual action
    def __init__(self, position, parent=None, depth=0, shooting = None, bullets = None, chickens =None, cost=1):
        self.parent = parent
        self.position=position
        self.depth = depth
        self.cost = cost
        self.shooting = shooting
        self.bullets = bullets if bullets is not None else []
        self.chickens = chickens if chickens is not None else [] 

    def __lt__(self, other):
        return self.cost < other.cost  


#searching algorithms function to explore the game:

def bfs(game_env, start_x, target_x):
    queue = deque([Node(start_x, None)])  
    visited = set()
    visited.add(start_x)
    global maxFrontierBFS 

    while queue:
        current_node = queue.popleft()
        global stepsBFS
        stepsBFS += 1 
        #print(f"Processing node at position {current_node.position}")  

        if current_node.position == target_x:
            path = []
            while current_node.parent is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            path.reverse()
            #print(f"BFS Steps: {steps}")  
            return path

        for dx in [-5, 5]:
            new_position = current_node.position + dx
            if 0 <= new_position <= game_env.SCREEN_WIDTH - 60 and new_position not in visited:
                queue.append(Node(new_position, current_node))
                visited.add(new_position)

        if len(queue) > maxFrontierBFS:
            maxFrontierBFS = len(queue)

    #print(f"BFS Steps: {stepsBFS}")  
    return []

def ucs(game_env, start_x, target_x):
    pq = []
    heapq.heappush(pq, (0, Node(start_x, None)))  
    visited = set()
    global maxFrontierUCS

    while pq:
        cumulative_cost, current_node = heapq.heappop(pq)
        global stepsUCS
        stepsUCS += 1 

        if current_node.position == target_x:
            path = []
            while current_node.parent is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            path.reverse()
            return path

        if current_node.position in visited:
            continue
        visited.add(current_node.position)

        for dx in [-5, 5]:
            new_position = current_node.position + dx
            if 0 <= new_position <= game_env.SCREEN_WIDTH - 60 and new_position not in visited:
                heapq.heappush(pq, (cumulative_cost + 1, Node(new_position, current_node)))
        print(len(pq))

        if len(pq) > maxFrontierUCS:
            maxFrontierUCS = len(pq)
            

    return []


def dfs(game_env, start_x, target_x):
    global maxFrontierDFS
    stack = [Node(start_x, None)]  
    visited = set()
    visited.add(start_x)
    
    while stack:
        current_node = stack.pop()
        global stepsDFS
        stepsDFS += 1  

        if current_node.position == target_x:
            path = []
            while current_node.parent is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            path.reverse()
            return path

        for dx in [-5, 5]:
            new_position = current_node.position + dx
            if 0 <= new_position <= game_env.SCREEN_WIDTH - 60 and new_position not in visited:
                stack.append(Node(new_position, current_node))
                visited.add(new_position)

        maxFrontierDFS = max(maxFrontierDFS, len(stack))

    return []

def ids(game_env, start_x, target_x):
    depth = 0
    while True:
        result = ids_search(game_env, start_x, target_x, depth)
        if result is not None:
            return result
        depth += 1

def ids_search(game_env, start_x, target_x, depth_limit):
    stack = deque([Node(start_x, None, 0)])  
    visited = set()
    visited.add(start_x)

    while stack:
        current_node = stack.pop()
        global maxFrontierIDS
        global stepsIDS
        stepsIDS += 1

        if current_node.position == target_x:
            path = []
            while current_node.parent is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            path.reverse()  
            return path

        if current_node.depth < depth_limit:
            for dx in [-5, 5]:
                new_position = current_node.position + dx
                if 0 <= new_position <= game_env.SCREEN_WIDTH - 60 and new_position not in visited:
                    stack.append(Node(new_position, current_node, current_node.depth + 1))
                    visited.add(new_position)
        maxFrontierIDS = max(maxFrontierIDS, len(stack))

    return None

def heuristic(position, chickens):  #heuristic based on distance
    return min(abs(position - chicken.x) for chicken in chickens)

#dawat el Manhattan diatnce
def astar(game_env, start_x, target_x):
    global maxFrontierAStar
    open_list = []
    heapq.heappush(open_list, Node(start_x, None, 0, 0))
    visited = set()
    visited.add(start_x)
    
    while open_list:
        global stepsAStar
        stepsAStar += 1
        current_node = heapq.heappop(open_list)

        if current_node.position == target_x:
            path = []
            while current_node.parent is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            path.reverse()
            return path

        for dx in [-5, 5]:
            new_position = current_node.position + dx
            if 0 <= new_position <= game_env.SCREEN_WIDTH - 60 and new_position not in visited:
                new_cost = current_node.cost + 1  
                new_h_cost = heuristic(new_position, game_env.chickens)
                new_node = Node(new_position, current_node, current_node.depth + 1, new_cost)
                heapq.heappush(open_list, new_node)
                visited.add(new_position)
        maxFrontierAStar = max(maxFrontierAStar, len(open_list))

    return []


#dawat el remaining chickens 
def astar2(game_env, start_x, target_x):
    global maxFrontierAStar2
    open_list = []
    heapq.heappush(open_list, Node(start_x, None, 0, 0))
    visited = set()
    visited.add(start_x)
    
    while open_list:
        global stepsAStar2
        stepsAStar2 += 1
        current_node = heapq.heappop(open_list)

        if current_node.position == target_x:
            path = []
            while current_node.parent is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            path.reverse()
            return path

        for dx in [-5, 5]:
            new_position = current_node.position + dx
            if 0 <= new_position <= game_env.SCREEN_WIDTH - 60 and new_position not in visited:
                new_cost = current_node.cost + 1  
                new_h_cost = count_remaining_chickens(game_env)  
                new_node = Node(new_position, current_node, current_node.depth + 1, new_cost + new_h_cost)
                heapq.heappush(open_list, new_node)
                visited.add(new_position)
        maxFrontierAStar2 = max(maxFrontierAStar2, len(open_list))

    return []

#da el Manhattan distance heristic
def greedy_search(game_env, start_x, target_x):
    open_list = []
    heapq.heappush(open_list, (abs(start_x - target_x), Node(start_x, None)))  
    visited = set()
    visited.add(start_x)

    while open_list:
        current_cost, current_node = heapq.heappop(open_list)
        global stepsGreedy
        stepsGreedy += 1  

        if current_node.position == target_x:
            path = []
            while current_node.parent is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            path.reverse()  
            #print(f"Greedy Search Steps: {stepsGreedy}") 
            return path

        for dx in [-5, 5]:
            new_position = current_node.position + dx
            if 0 <= new_position <= game_env.SCREEN_WIDTH - 60 and new_position not in visited:
                new_cost = abs(new_position - target_x)
                heapq.heappush(open_list, (new_cost, Node(new_position, current_node)))
                visited.add(new_position)

    #print(f"Greedy Search Steps: {stepsGreedy}")  
    return [] 

def count_remaining_chickens(g): #heristic function on remaining chiken count, that is best when close to zero
    return len(g.chickens)

#da el remaining chickens heristic
def greedyheuristic2(game_env, start_x, target_x):
    open_list = []
    heapq.heappush(open_list, (count_remaining_chickens(game_env), Node(start_x, None))) 
    visited = set()
    visited.add(start_x)

    while open_list:
        current_cost, current_node = heapq.heappop(open_list)
        
        global stepsGreedy2
        stepsGreedy2 += 1  

        if current_node.position == target_x:
            path = []
            while current_node.parent is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            path.reverse()  
            return path

        for dx in [-5, 5]:
            new_position = current_node.position + dx
            if 0 <= new_position <= game_env.SCREEN_WIDTH - 60 and new_position not in visited:
                new_cost = count_remaining_chickens(game_env) 
                heapq.heappush(open_list, (new_cost, Node(new_position, current_node)))
                visited.add(new_position)

    return []  


#awyaaa climb the hillll
def hill_climbing(game_env, start_x, target_x):
    current_position = start_x

    while current_position != target_x:
        global stepsHill
        stepsHill += 1  

        neighbors = [current_position - 5, current_position + 5]
        valid_neighbors = [pos for pos in neighbors if 0 <= pos <= game_env.SCREEN_WIDTH - 60]
        if not valid_neighbors:
            break

        next_position = min(valid_neighbors, key=lambda pos: abs(pos - target_x))
        print(f"Step {stepsHill}: Current Position = {current_position}, Target = {target_x}, Next = {next_position}")
        if abs(next_position - target_x) >= abs(current_position - target_x):
            break
        current_position = next_position

    return [current_position]  



def simulated_annealing(game_env, start_x, target_x, schedule=None):
    if schedule is None:
        def schedule(t):
            initial_temp = 100  
            decay_rate = 0.01   
            return initial_temp / (1 + decay_rate * t)  
    current_state = start_x  
    current_value = abs(current_state - target_x)  
    path = [current_state]  

    for t in range(1, 1000):  



        T = schedule(t) 
        global stepsAnnealing
        stepsAnnealing += 1   
        if T == 0 or current_value == 0:  
            break
        



        next_states = [current_state - SHIP_SPEED, current_state + SHIP_SPEED]      
        next_state = random.choice(next_states)
        if next_state < 0 or next_state > game_env.SCREEN_WIDTH - 60:
            continue
        next_value = abs(next_state - target_x)  
        delta = current_value - next_value  
        if delta > 0 or random.random() < math.exp(delta / T):
            current_state = next_state
            current_value = next_value
            path.append(current_state)  
    return path


def temperature_schedule(t):  #temperature function for simulated annealing to measure temperature
    
    initial_temp = 100 
    decay_rate = 0.01  
    return initial_temp / (1 + decay_rate * t)  

#helper functions to implement genetic algorithm:

def fitness_function(node):
    if not node.chickens:
        return float('-inf')  

    closest_chicken_distance = min(abs(node.position - chicken.x) for chicken in node.chickens)
    effective_shots = sum(1 for bullet in node.bullets if any(bullet.colliderect(chicken) for chicken in node.chickens))
    return -closest_chicken_distance + 10 * effective_shots
    
def create_population(size, lower_bound, upper_bound, game_env):
    return [
        Node(
            random.uniform(lower_bound, upper_bound),
            shooting=random.choice([True, False]),
            bullets=game_env.bullets.copy(),
            chickens=game_env.chickens.copy() 
        )
        for _ in range(size)
    ]

def select_population(population, fitnesses, tournament_size=3):
    if len(population) < tournament_size:
        tournament_size = len(population)  

    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected
def crossover(parent1, parent2):
    alpha = random.random()
    child_position = alpha * parent1.position + (1 - alpha) * parent2.position
    child_shooting = random.choice([parent1.shooting, parent2.shooting]) 
    return Node(child_position, child_shooting, parent1.bullets, parent1.chickens)
def mutate(individual, mutation_rate, lower_bound, upper_bound):
    if random.random() < mutation_rate:
        mutation = random.uniform(-10, 10)
        individual.position = max(lower_bound, min(upper_bound, individual.position + mutation))
        individual.shooting = random.choice([True, False]) 
    return individual

def genetic_algorithm(game_env, generations=10, population_size=20, mutation_rate=0.1):
    global stepsGenetic
    lower_bound, upper_bound = 0, game_env.SCREEN_WIDTH - 60
    population = create_population(population_size, lower_bound, upper_bound, game_env)

    for generation in range(generations):
        stepsGenetic +=1
        fitnesses = [fitness_function(ind) for ind in population]

        population = select_population(population, fitnesses)

        next_population = []
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[(i + 1) % len(population)]
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate, lower_bound, upper_bound)
            next_population.append(child)

        population = next_population

    best_individual = max(population, key=lambda ind: fitness_function(ind))
    return best_individual



def Q_learning(game_env):
    SCREEN_HEIGHT = game_env.SCREEN_HEIGHT
    SCREEN_WIDTH = game_env.SCREEN_WIDTH
    ship_y_Q = SCREEN_HEIGHT - 70
    actions = ["right", "left", "shoot"]
    Q_table = {}
    max_position = 290
    min_position = 90
    midway = None

    def get_starting_location():
        ship_x_Q = random.choice([90, 140, 190, 240, 290])
        return ship_x_Q

    root_node = Node(position=get_starting_location(), bullets=[], chickens=[pygame.Rect(90, 50, 40, 40), pygame.Rect(140, 50, 40, 40), pygame.Rect(190, 50, 40, 40), pygame.Rect(240, 50, 40, 40), pygame.Rect(290, 50, 40, 40)])

    def is_terminal_state(node):
        return len(node.chickens) == 0

    
    def find_final_path():
        path = []
        reward = 0 
        state = root_node
        i = 0
        while not is_terminal_state(state) and i < 100:
            if state.position not in Q_table:
                Q_table[state.position] = {action: 0.0 for action in actions}
            action_index = np.argmax(list(Q_table[state.position].values()))
            path.append(actions[action_index])
            
            new_state = apply_action(state, action_index)
            
            # Calculate the reward for the action taken
            action = actions[action_index]
            reward += calculate_reward(action, state)  # Accumulate reward
            
            state = new_state  
            i += 1
        return path, reward


    def get_next_action(node, epsilon):
        state = node.position
        if state not in Q_table:
            Q_table[state] = {action: 0.0 for action in actions}
        if np.random.random() < epsilon:
            return np.random.randint(3) 
        else:
            return np.argmax(Q_table[state])  

    def apply_action(current_node, action_index):
        new_node = Node(position=current_node.position, bullets=current_node.bullets.copy(), chickens=current_node.chickens.copy())
        if actions[action_index] == 'right' and new_node.position < 290:
            new_node.position += 50
        elif actions[action_index] == 'left' and new_node.position > 90:
            new_node.position -= 50
        elif actions[action_index] == 'shoot':
            game_env.shoot()
            new_node.bullets.append(pygame.Rect(new_node.position + 30, game_env.ship_y, 5, 10))
        return new_node

    def calculate_reward(action, current_node):
        if action == "shoot" and any(chicken.colliderect(bullet) for bullet in current_node.bullets for chicken in current_node.chickens):
            return 10
        elif action == "shoot":
            return -5
        else:
            return -1

    for episode in range(100):
        current_node = root_node
        steps = 0
        while not is_terminal_state(current_node) and steps < 1000:
            steps += 1
            state = current_node

            action_index = get_next_action(state, epsilon=0.1)
            next_node = apply_action(state, action_index)

            reward = calculate_reward(actions[action_index], state)
            next_state = next_node.position
            if next_state not in Q_table:
                Q_table[next_state] = {action: 0.0 for action in actions}

            Q_table[state.position][actions[action_index]] += 0.1 * (reward + 0.9 * max(Q_table[next_state].values()) - Q_table[state.position][actions[action_index]])

            current_node = next_node

        if episode == 49:
            midway = {state: q.copy() for state, q in Q_table.items()}
        
    print("\nMidway Q-values (Episode 50):")
    for state, values in midway.items():
        print(f"State {state}: {values}")

    print("\nFinal Q-values (Episode 100):")
    for state, values in Q_table.items():
        print(f"State {state}: {values}")
    print("\nPolicy Learned:")
    for state in Q_table:
        best_action = max(Q_table[state], key=Q_table[state].get)
        print(f"State {state}: Best action is {best_action}")
        # Print Q-table at each episode
        print(f"Episode {episode + 1}: Q-table")
        for state in Q_table:
            print(f"State {state}: {Q_table[state]}")

    # Print final best path
    best_path, reward = find_final_path()
    print(f"Best Path: {best_path}")
    print(f"\n\n\nReward is {reward}")
        


#simulator function that simulates the solution of each path in the algorithm usung a gui
def simulate_game(game_env, algorithm, algorithm_name): 
    actions_taken = []

    while game_env.chickens:
        game_env.screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        font = pygame.font.SysFont("Arial", 24)
        algorithm_text = font.render(f"Algorithm: {algorithm_name}", True, (255, 255, 255))
        game_env.screen.blit(algorithm_text, (10, 10))

        if game_env.chickens:
            current_chicken = min(
                game_env.chickens,
                key=lambda chicken: abs(chicken.x + 20 - (game_env.ship_x + 30))
            )
            target_x = current_chicken.x + 40 // 2 - 60 // 2  

            path_to_follow = algorithm(game_env, game_env.ship_x, target_x)

            for next_position in path_to_follow:
                if game_env.ship_x < next_position:
                    game_env.ship_x += 5
                    actions_taken.append('move_right')
                elif game_env.ship_x > next_position:
                    game_env.ship_x -= 5
                    actions_taken.append('move_left')

            if game_env.ship_x == target_x and game_env.active_bullet is None:
                game_env.active_bullet = pygame.Rect(game_env.ship_x + 60 // 2 - 2, game_env.ship_y, 4, 10)
                actions_taken.append('fire_bullet')

        if game_env.active_bullet:
            game_env.active_bullet.y -= 5
            if game_env.active_bullet.y < 0:
                game_env.active_bullet = None

            for chicken in game_env.chickens[:]:
                if game_env.active_bullet.colliderect(chicken):
                    game_env.chickens.remove(chicken)
                    game_env.active_bullet = None
                    break

        game_env.draw_spaceship()
        game_env.draw_chickens()
        game_env.draw_bullet()

        pygame.display.flip()
        pygame.time.delay(1)

    return actions_taken

#simulator function that simulates the genetic solution of each path in the algorithm using a gui
def simulate_genetic(game_env, generations=10, population_size=20, mutation_rate=0.1):
    clock = pygame.time.Clock()
    game_running = True

    while game_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        best_individual = genetic_algorithm(game_env, generations=generations, population_size=population_size, mutation_rate=mutation_rate)

        font = pygame.font.SysFont("Arial", 24)
        algorithm_text = font.render("Algorithm: Genetic Algorithm", True, (255, 255, 255))
        game_env.screen.blit(algorithm_text, (10, 10))

        game_env.ship_x = best_individual.position

        if best_individual.shooting:
            game_env.shoot()

        for bullet in game_env.bullets[:]:
            bullet.y -= 5  
            if bullet.y < 0:
                game_env.bullets.remove(bullet)  

        for bullet in game_env.bullets[:]:
            for chicken in game_env.chickens[:]:
                if bullet.colliderect(chicken):
                    game_env.chickens.remove(chicken)
                    game_env.bullets.remove(bullet)
                    break

        game_env.screen.fill(BLACK)
        game_env.draw_spaceship()
        game_env.draw_chickens()
        for bullet in game_env.bullets: 
            pygame.draw.rect(game_env.screen, (255, 0, 0), bullet)

        if not game_env.chickens:
            game_running = False

        pygame.display.flip()
        clock.tick(60)


#main function calling each algorithm and displays comparison in console
#finally kolo workingggg :)
#wadyyyyyyyyyy
def main():
    game_env = GameEnvironment()

    Q = Q_learning(game_env)
    actions = simulate_game(game_env, Q,'BFS')  

    # game_env.reset()
    # t0 = time.perf_counter()
    # actions = simulate_game(game_env, bfs,'BFS')  
    # t1 = time.perf_counter()
    # timeBFS = t1-t0
    # print(actions) 

    # game_env.reset()
    # t0 = time.perf_counter()
    # actions = simulate_game(game_env, dfs,"DFS")  
    # t1 = time.perf_counter()
    # timeDFS = t1-t0
    # print(actions)
    
    # game_env.reset()
    # t0 = time.perf_counter()
    # actions = simulate_game(game_env, ids, 'IDS')
    # t1 = time.perf_counter()
    # timeIDS = t1-t0
    # print(actions)

    # game_env.reset()
    # t0 = time.perf_counter()
    # actions = simulate_game(game_env, ucs,'UCS')
    # t1 = time.perf_counter()
    # timeUCS = t1-t0
    # print(actions)   

    # game_env.reset()
    # t0 =time.perf_counter()
    # actions = simulate_game(game_env, greedy_search,'Greedy')  
    # t1 = time.perf_counter()
    # timeGreedy = t1-t0
    # print(actions)

    # game_env.reset()
    # t0 =time.perf_counter()
    # actions = simulate_game(game_env, greedyheuristic2,'Greedy 2')  
    # t1 = time.perf_counter()
    # timeGreedy2 = t1-t0
    # print(actions)

    
    # game_env.reset()
    # t0 =time.perf_counter()
    # actions = simulate_game(game_env, astar,'A* ')  
    # t1 = time.perf_counter()
    # timeAStar = t1-t0
    # print(actions)

    # game_env.reset()
    # t0 =time.perf_counter()
    # actions = simulate_game(game_env, astar2,'A* 2')  
    # t1 = time.perf_counter()
    # timeAStar2 = t1-t0
    # print(actions)
    
    # game_env.reset()
    # t0 =time.perf_counter()
    # actions = simulate_game(game_env, hill_climbing,"Hill Climbing")
    # t1 = time.perf_counter()
    # timeHill = t1-t0
    # print(actions)

    # game_env.reset()
    # t0 =time.perf_counter()
    # actions = simulate_game(game_env, simulated_annealing,"Simulated Annealing") 
    # t1 = time.perf_counter()
    # timeSimualated = t1-t0
    # print(actions)


    # game_env.reset()
    # t0 =time.perf_counter()
    # simulate_genetic(game_env)
    # t1 = time.perf_counter()
    # timeGenetic = t1-t0
    # print(actions)

    
    
    # print(f"BFS steps: {stepsBFS} and time:{timeBFS} and max frontier:{maxFrontierBFS}")
    # print(f"DFS steps: {stepsDFS} and time:{timeDFS} and max frontier:{maxFrontierDFS}") 
    # print(f"IDS steps: {stepsIDS} and time:{timeIDS} and max frontier:{maxFrontierIDS}")
    # print(f"UCS steps: {stepsUCS} and time:{timeUCS} and max frontier:{maxFrontierUCS}")
    # print(f"Greddy steps: {stepsGreedy} and time:{timeGreedy}")# and max frontier:{maxFrontierGreedy}") 
    # print(f"Greddy Heuristic 2 steps: {stepsGreedy} and time:{timeGreedy2}")# and max frontier:{maxFrontierGreedy}") 
    # print(f"A* steps: {stepsAStar} and time:{timeAStar} and max frontier:{maxFrontierAStar}") 
    # print(f"A* steps: {stepsAStar2} and time:{timeAStar2} and max frontier:{maxFrontierAStar2}") 
    # print(f"Hill climbing steps: {stepsHill} and time:{timeHill}")# and max frontier:{maxFrontierHill}")  
    # print(f"Simulated Annealing steps: {stepsAnnealing} and time:{timeSimualated}")# and max frontier:{maxFrontierHill}")  
    # print(f"Genetic Algorithm steps: {stepsGenetic} and time:{timeGenetic}")# and max frontier:{maxFrontierHill}")  

if __name__ == "__main__":
    main()
