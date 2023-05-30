import math
import random
import numpy as np

class Task:
    def __init__(self, duration, weight, deadline):
        self.duration = duration
        self.weight = weight
        self.deadline = deadline
        self.finishedTime = duration

    def update_finished_time(self, previousTaskFinishedAt):
        self.finishedTime += previousTaskFinishedAt

    def calculate_tardiness(self, previousTask):
        if previousTask is None:
            return np.zeros_like(self.finishedTime)  # No previous task, so no tardiness
        else:
            return np.maximum(0, previousTask.finishedTime - self.deadline)

    def calculate_weighted_tardiness(self, previousTask=None):
        if previousTask is None:
            return self.duration * self.weight
        else:
            tardiness = self.calculate_tardiness(previousTask)
            try:
                weighted_tardiness = np.multiply(tardiness, self.weight)
            except OverflowError:
                weighted_tardiness = float('inf')  # lub inna wartość oznaczająca przepełnienie
            return weighted_tardiness

class PotentialSolution:
    def __init__(self, tasks_in_order):
        self.tasks_in_order = tasks_in_order
        self.counter = 0
        self.fitness_value = self.calculate_fitness()
        self.roulette_prob = -1

    def calculate_roulette_probability(self, otherSolutions):
        total_inverse_fitness = sum(1 / solution.fitness_value for solution in otherSolutions)
        return np.array([(1 / solution.fitness_value) / total_inverse_fitness for solution in otherSolutions])

    def calculate_fitness(self):
        fitness_value = 0
        previous_task = None  # Initialize the previous task
        for task in self.tasks_in_order:
            fitness_value = np.add(fitness_value, task.calculate_weighted_tardiness(previous_task))
            if previous_task is not None:
                task.update_finished_time(previous_task.finishedTime)  # Update the finished time of the current task
            previous_task = task  # Set the current task as the previous task for the next iteration
        return fitness_value

    def replace_solution(self, solutionGenerator):
        self.counter = 0
        self.tasks_in_order = solutionGenerator.permutate(self.tasks_in_order)
        self.fitness_value = self.calculate_fitness()

    def get_better_solution(self, solution, without_change_limit, solutionGenerator):
        if self.fitness_value > solution.fitness_value:
            return self
        else:
            self.incrementCounter(without_change_limit, solutionGenerator)
            return solution

    def incrementCounter(self, without_change_limit, solutionGenerator):
        if self.counter < without_change_limit:
            self.counter += 1
        else:
            self.replace_solution(solutionGenerator)

class SolutionGenerator:
    def generate_random_solutions(self, tasks, no_solutions):
        solutions = []
        for _ in range(no_solutions):
            indices = np.random.permutation(len(tasks))
            solution = PotentialSolution([tasks[i] for i in indices])
            solutions.append(solution)
        return solutions

    def permutate(self, tasks):
        return np.random.permutation(tasks)

    def generate_neighbor(self, solution, num_changes):
        tasks = np.array(solution.tasks_in_order)  # Convert tasks list to a NumPy array
        for _ in range(num_changes):
            indices = np.random.choice(len(tasks), 2, replace=False)
            tasks[indices] = tasks[indices[::-1]]  # Reverse the order of selected tasks
        return PotentialSolution(tasks.tolist())

class ABC:
    def __init__(self, population_size, no_iterations, without_change_limit, tasks_data, generator):
        self.population_size = population_size
        self.no_iterations = no_iterations
        self.without_change_limit = without_change_limit
        self.tasks_data = tasks_data
        self.generator = generator

    def run(self):
        tasks = self.create_tasks()
        solutions = self.generator.generate_random_solutions(tasks, self.population_size)
        best_solution = self.get_best_solution(solutions)

        for _ in range(self.no_iterations):
            employed_bees = self.generate_employed_bees(solutions)
            onlooker_bees = self.generate_onlooker_bees(employed_bees)
            best_solution = self.get_best_solution(solutions + employed_bees + onlooker_bees)
            solutions = self.generate_scout_bees(best_solution)

        return best_solution

    def create_tasks(self):
        tasks = []
        for task_data in self.tasks_data:
            task = Task(task_data['duration'], task_data['weight'], task_data['deadline'])
            tasks.append(task)
        return tasks

    def generate_employed_bees(self, solutions):
        employed_bees = []
        for solution in solutions:
            num_changes = math.floor(random.uniform(0, 1) * len(solution.tasks_in_order)) + 1
            neighbor_solution = self.generator.generate_neighbor(solution, num_changes)
            better_solution = solution.get_better_solution(neighbor_solution, self.without_change_limit, self.generator)
            employed_bees.append(better_solution)
        return employed_bees

    def generate_onlooker_bees(self, employed_bees):
        onlooker_bees = []
        for solution in employed_bees:
            roulette_probability = solution.calculate_roulette_probability(employed_bees)
            selected_solution = np.random.choice(employed_bees, p=roulette_probability)
            num_changes = math.floor(random.uniform(0, 1) * len(selected_solution.tasks_in_order)) + 1
            neighbor_solution = self.generator.generate_neighbor(selected_solution, num_changes)
            better_solution = solution.get_better_solution(neighbor_solution, self.without_change_limit, self.generator)
            onlooker_bees.append(better_solution)
        return onlooker_bees

    def generate_scout_bees(self, best_solution):
        scout_bees = []
        if best_solution.counter >= self.without_change_limit:
            new_solution = PotentialSolution(self.generator.permutate(best_solution.tasks_in_order))
            scout_bees.append(new_solution)
        else:
            scout_bees.append(best_solution)
        return scout_bees

    def get_best_solution(self, solutions):
        return max(solutions, key=lambda x: x.fitness_value)


def generate_instance(n, seed):
    random.seed(seed)
    tasks = []

    # Oblicz sumę wartości pi
    sum_p = 0
    for i in range(1, n + 1):
        pi = random.randint(1, 30)
        sum_p += pi

    # Generuj zadania
    for i in range(1, n + 1):
        pi = random.randint(1, 30)
        wi = random.randint(1, 30)
        di = random.randint(1, sum_p)

        task = {'duration': pi, 'weight': wi, 'deadline': di}
        tasks.append(task)

    return tasks


if __name__ == '__main__':
    no_tasks = [2, 3, 4, 5]
    for i in no_tasks:
        tasks = generate_instance(i, 123)
        generator = SolutionGenerator()
        abc = ABC(10, 10, 3, tasks, generator)
        best_solution = abc.run()
        print("Best solution:")
        for task in best_solution.tasks_in_order:
            print("Duration:", task.duration, "| Weight:", task.weight, "| Deadline:", task.deadline,
                  "| Finished Time:", task.finishedTime)

























