import pygame
import random
import numpy as np
import sys

# Initialize Pygame
pygame.init()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BULLET_SPEED = 35
BULLET_COOLDOWN = 500

# Game environment responsible for all game aspects
class GameEnvironment:
    def __init__(self, width=800, height=600):
        pygame.init()
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

    def create_chickens(self):
        chickens = []
        for i in range(5):  
            for j in range(5):
                if random.random() > 0.55:
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
            pygame.draw.rect(self.screen, RED, self.active_bullet)

    def shoot(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_shot_time > BULLET_COOLDOWN:
            bullet = pygame.Rect(self.ship_x + 30, self.ship_y, 5, 10)
            self.bullets.append(bullet)
            self.last_shot_time = current_time

    def get_state(self): 
        chickens_positions = tuple((chicken.x, chicken.y) for chicken in self.chickens) 
        bullet_position = (self.active_bullet.x, self.active_bullet.y) if self.active_bullet else (None, None) 
        return (self.ship_x, chickens_positions, bullet_position)

class Node:
    def __init__(self, position, parent=None, depth=0, shooting=None, bullets=None, chickens=None, cost=1):
        self.parent = parent
        self.position = position
        self.depth = depth
        self.cost = cost
        self.shooting = shooting
        self.bullets = bullets if bullets is not None else []
        self.chickens = chickens if chickens is not None else []

    def __lt__(self, other):
        return self.cost < other.cost

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
            return 1
        elif action == "shoot":
            return -0.001
        else:
            return -1

    for episode in range(100):
        current_node = root_node
        steps = 0
        while not is_terminal_state(current_node) and steps < 1000:
            steps += 1
            state = current_node

            action_index = get_next_action(state, epsilon=0.5)
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

    return Q_table



def simulate_game(game_env, Q_table, algorithm_name):
    clock = pygame.time.Clock()

    while game_env.chickens:
        game_env.screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        state = game_env.get_state()
        action = max(Q_table.get(state, {'shoot': 0, 'left': 0, 'right': 0}).items(), key=lambda x: x[1])[0]

        if action == "right" and game_env.ship_x < game_env.SCREEN_WIDTH - 60:
            game_env.ship_x += 5
        elif action == "left" and game_env.ship_x > 0:
            game_env.ship_x -= 5
        elif action == "shoot":
            game_env.shoot()

        game_env.draw_spaceship()
        game_env.draw_chickens()
        game_env.draw_bullet()

        font = pygame.font.SysFont("Arial", 24)
        texts = [
            f"Algorithm: {algorithm_name}",
            f"Chickens: {len(game_env.chickens)}",
            f"Action: {action}"
        ]
        
        for i, text in enumerate(texts):
            text_surface = font.render(text, True, WHITE)
            game_env.screen.blit(text_surface, (10, 10 + i * 25))

        pygame.display.flip()
        clock.tick(60)

def main():
    game_env = GameEnvironment()
    Q_table = Q_learning(game_env)
    simulate_game(game_env, Q_table, "Q-Learning")

if __name__ == "__main__":
    main()

