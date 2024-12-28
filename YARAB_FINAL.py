import pygame
import random
import numpy as np
import sys

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
        if current_time - self.last_shot_time > 500:  # BULLET_COOLDOWN
            bullet = pygame.Rect(self.ship_x + 30, self.ship_y, 5, 10)
            self.bullets.append(bullet)
            self.last_shot_time = current_time

    def calculate_reward(self, action, hit_chicken=False):
        reward = 0
        if action == 'fire_bullet':
            reward = 0.01 
        
        if hit_chicken:
            reward += 10  
        
        return reward


class QLearningAgent:
    def __init__(self, state_space, action_space, learning_rate=0.3, discount_factor=0.9, exploration_rate=0.7, exploration_decay=0.995):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = np.zeros(state_space + (len(action_space),))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(range(len(self.action_space)))
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state][action]
        target = reward + self.discount_factor * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (target - predict)

    def update_exploration_rate(self):
        self.exploration_rate *= self.exploration_decay

    def get_policy(self):
        policy = {}
        for ship_x in range(self.state_space[0]):  
            state = (ship_x,)
            best_action_index = np.argmax(self.q_table[state])  
            policy[state] = self.action_space[best_action_index]
        return policy


def main():
    pygame.init()
    game_env = GameEnvironment()

    state_space = (game_env.SCREEN_WIDTH // 10,)
    action_space = ['move_left', 'move_right', 'fire_bullet']
    agent = QLearningAgent(state_space, action_space)

    total_training_episodes = 10 
    rewards_per_episode = []

    for episode in range(total_training_episodes):
        game_env.reset()
        total_reward = 0
        steps_in_episode = 0  

        while game_env.chickens:
            state = (game_env.ship_x // 10,)

            action_index = agent.choose_action(state)
            action = action_space[action_index]

            if action == 'move_left':
                game_env.ship_x = max(0, game_env.ship_x - 10)
            elif action == 'move_right':
                game_env.ship_x = min(game_env.SCREEN_WIDTH - 60, game_env.ship_x + 10)
            elif action == 'fire_bullet' and game_env.active_bullet is None:
                game_env.active_bullet = pygame.Rect(game_env.ship_x + 30, game_env.ship_y, 4, 10)


            hit_chicken = False  

            if game_env.active_bullet:
                game_env.active_bullet.y -= 5
                if game_env.active_bullet.y < 0:
                    game_env.active_bullet = None

                for chicken in game_env.chickens[:]:
                    if game_env.active_bullet and game_env.active_bullet.colliderect(chicken):
                        game_env.chickens.remove(chicken)
                        game_env.active_bullet = None
                        hit_chicken = True  


            reward = game_env.calculate_reward(action, hit_chicken)
            total_reward += reward
            next_state = (game_env.ship_x // 10,)

 
            agent.learn(state, action_index, reward, next_state)

            game_env.screen.fill((0, 0, 0))
            game_env.draw_spaceship()
            game_env.draw_chickens()
            game_env.draw_bullet()
            pygame.display.flip()

            steps_in_episode += 1 

        rewards_per_episode.append(total_reward)  
        agent.update_exploration_rate()

        print(f"Episode {episode + 1}/{total_training_episodes}, Steps: {steps_in_episode}, Total Reward: {total_reward}")
 
        if episode == 5:
            print("Midway Q-values:")
            print(agent.q_table)
        if episode == total_training_episodes - 1:
            print("Final Q-values:")
            print(agent.q_table)

    policy = agent.get_policy()
    print("Learned Policy:")
    for state, action in policy.items():
        print(f"State {state}: Action '{action}'")

    pygame.quit()
    print("Training complete. Rewards per episode:", rewards_per_episode)

if __name__ == "__main__":
    main()
