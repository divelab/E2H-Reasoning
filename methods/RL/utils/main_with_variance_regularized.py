# This is a modified section of main.py that includes the variance regularized scheduler
# Copy this content to replace the corresponding section in main.py

# Add this import at the top with other imports:
from variance_regularized_scheduler import _variance_regularized_schedule, update_variance_regularized_performance, reset_variance_regularized_state

# Replace the TaskSampler class (starting around line 160) with this version:

class TaskSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_tasks, total_iterations, data_schedule, batch_size, scheduler_params, seed=0):
        """
        Args:
          dataset: a HF dataset; each sample is assumed to be a dict including "task" (an integer 0 to num_tasks-1)
          num_tasks: total number of task categories (e.g. 4)
          total_iterations: total training iterations (T)
          current_iter_fn: callable that returns current iteration (t)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_dataset_len = len(self.dataset)
        self.num_tasks = num_tasks
        self.total_iterations = total_iterations
        self.rng = np.random.default_rng(seed)
        task_col = np.array(self.dataset['task'])
        self.indices_by_task = {
            t: self.rng.permutation(np.where(task_col == t)[0])
            for t in range(num_tasks)
        }
        self.schedule_funcs = {
            'balanced': self._balanced_schedule,
            'cosine': self._cosine_schedule,
            'gaussian': partial(self._gaussian_schedule, **scheduler_params),
            'classic': self._step_schedule,
            'variance_regularized': partial(_variance_regularized_schedule, **scheduler_params)
        }
        log_on_main(f"Data Schedule: {data_schedule}")
        self.schedule_func = self.schedule_funcs[data_schedule]
        self.data_schedule = data_schedule  # Store for later use
        
        # Reset variance regularized state if using that scheduler
        if data_schedule == 'variance_regularized':
            reset_variance_regularized_state()
    
    # Classical Curriculum Learning
    @staticmethod    
    def _step_schedule(t, T, num_tasks):
        active_task = min(int(t * num_tasks / T), num_tasks - 1)
        return dict(enumerate(np.eye(num_tasks)[active_task].tolist()))

    def __iter__(self):
        task_ptrs = {t: 0 for t in range(self.num_tasks)}
        indices_by_task = {t: idx.copy() for t, idx in self.indices_by_task.items()}
        
        for i in range(self.total_iterations):
            probs_dict = self.schedule_func(i, self.total_iterations, self.num_tasks)

            probs = np.array([probs_dict[j] for j in range(self.num_tasks)])
            # Sample a task for each slot in the batch using the probabilities.
            chosen_tasks = np.random.choice(np.arange(self.num_tasks), size=self.batch_size, p=probs, replace=True)
            batch_indices = []

            for task in chosen_tasks:
                indices = self.indices_by_task[task]
                ptr = task_ptrs[task]
                if ptr >= len(indices):
                    # Once exhausted, reshuffle that task's pool
                    indices = self.rng.permutation(indices)
                    indices_by_task[task] = indices
                    ptr = 0
                batch_indices.append(int(indices[ptr]))
                task_ptrs[task] = ptr + 1
                
            # Store batch indices and tasks for performance tracking
            self.last_batch_indices = batch_indices
            self.last_chosen_tasks = chosen_tasks
            
            log_on_main(f"Iteration {i}: Batch indices: {batch_indices}: Task Difficulties: {chosen_tasks}")
            yield from batch_indices

    def __len__(self):
        return self.total_iterations * self.batch_size
    
    @staticmethod
    def _balanced_schedule(t, T, num_tasks):
        return {i: 1. / num_tasks for i in range(num_tasks)}

    @staticmethod
    def _cosine_schedule(t, T, num_tasks):
        total = num_tasks * (num_tasks + 1) / 2.0
        early = {i: (num_tasks - i) / total for i in range(num_tasks)}
        late = {i: (i + 1) / total for i in range(num_tasks)}
        alpha = 0.5 * (1 + math.cos(math.pi * t / T))
        probs = {i: alpha * early[i] + (1 - alpha) * late[i] for i in range(num_tasks)}
        # Enforce symmetric floor equal to the minimum probability in early/late.
        p_min = 2 / (num_tasks * (num_tasks + 1))
        for i in range(num_tasks):
            probs[i] = max(probs[i], p_min)
        norm = sum(probs.values())
        return {i: probs[i] / norm for i in probs}

    @staticmethod
    def _gaussian_schedule(t, T, num_tasks, mu_exp, sigma, min_prob: Union[bool, float]=False):
        '''
        Gaussian schedule for task sampling.
        mu_exp: exponent for the mean, typically 1.0. Move faster at the beginning: < 1.0. Move slower at the beginning: > 1.0
        sigma: standard deviation of the Gaussian distribution
        min_prob: minimum probability for each task
        '''
        # Move mean from 0 to (num_tasks-1) as time progresses, Use sqrt(t / T) to boost the the speed at the beginning
        mu = (t / T) ** mu_exp * (num_tasks - 1)
        p_min = (2 / (num_tasks * (num_tasks + 1))) if (min_prob is True) else (min_prob if isinstance(min_prob, float) else None)
        if p_min is None: raise ValueError("min_prob should be either a boolean or a float")
        if num_tasks * p_min > 1: raise ValueError("num_tasks * p_min must not exceed 1")
        
        # Compute normalized Gaussian probabilities.
        base = [math.exp(-((i - mu) ** 2) / (2 * sigma ** 2)) for i in range(num_tasks)]
        total = sum(base)
        q = [b / total for b in base]
        
        # Mix with uniform floor to guarantee each probability is at least p_min.
        return {i: p_min + (1 - num_tasks * p_min) * q_i for i, q_i in enumerate(q)}