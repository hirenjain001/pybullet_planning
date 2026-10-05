import time

from .primitives import extend_towards
from .rrt import TreeNode, configs
from .utils import irange, RRT_ITERATIONS, INF, elapsed_time

__all__ = [
    'rrt_connect',
    'birrt',
]


def _snapshot(nodes1, nodes2):
    """
    Create a shallow snapshot of the two trees.

    TreeNode objects themselves are not modified after creation;
    the tree lists are what grow. Therefore copying the lists
    preserves the tree state at this point in the algorithm while
    retaining parent pointers.
    """
    return list(nodes1), list(nodes2)


import time

from .primitives import extend_towards
from .rrt import TreeNode, configs
from .utils import irange, RRT_ITERATIONS, INF, elapsed_time

__all__ = [
    'rrt_connect',
    'birrt',
]


def rrt_connect(q1, q2, distance_fn, sample_fn, extend_fn, collision_fn,
                max_iterations=RRT_ITERATIONS, max_time=INF, verbose=False,
                draw_fn=None, enforce_alternate=False,
                return_history=False, ml_model=None, check_after=50, **kwargs):
    print("DEBUG rrt_connect max_iterations =", max_iterations)

    start_time = time.time()

    # Always create the two root nodes first.
    nodes1 = [TreeNode(q1)]
    nodes2 = [TreeNode(q2)]

    # t = 0: initial root-only state.
    history = []

    if return_history:
        history.append((list(nodes1), list(nodes2)))

    # Invalid start or goal configuration.
    # We still return the root-only history so the dataset can
    # contain this attempt as a negative example.
    # if collision_fn(q1) or collision_fn(q2):
    #     return (None, history) if return_history else None

    start_collision = collision_fn(q1)
    goal_collision = collision_fn(q2)

    print("DEBUG endpoint collision:", start_collision, goal_collision)

    if start_collision or goal_collision:
        return (None, history) if return_history else None

    for iteration in irange(max_iterations):

        print("DEBUG entered iteration:", iteration)


        if max_time <= elapsed_time(start_time):
            break

        # Optional ML early stopping.
        # Dataset generation should normally leave ml_model=None.
        if ml_model is not None and iteration > 0 and iteration % check_after == 0:
            is_feasible = ml_model.predict(nodes1, nodes2)

            if not is_feasible:
                if verbose:
                    print(
                        f"ML Model triggered early stop at iteration {iteration}"
                    )
                return (None, history) if return_history else None

        # Decide which tree is expanded first.
        if enforce_alternate:
            swap = iteration % 2
        else:
            swap = len(nodes1) > len(nodes2)

        tree1, tree2 = nodes1, nodes2

        if swap:
            tree1, tree2 = nodes2, nodes1

        # Sample a random target configuration.
        target = sample_fn()

        if draw_fn:
            draw_fn(target, [])

        # ---------------------------------------------------------
        # 1. Expand the first tree.
        # ---------------------------------------------------------
        last1, _ = extend_towards(
            tree1,
            target,
            distance_fn,
            extend_fn,
            collision_fn,
            swap,
            **kwargs
        )

        # ---------------------------------------------------------
        # Record the state BEFORE attempting the connection.
        #
        # This is critical:
        # the second tree's successful extension must NOT leak into
        # the snapshot that receives label=1.
        # ---------------------------------------------------------
        if return_history:
            if swap:
                history.append((list(tree2), list(tree1)))
            else:
                history.append((list(tree1), list(tree2)))

        # ---------------------------------------------------------
        # 2. Try to connect the second tree to the first tree.
        # ---------------------------------------------------------
        last2, success = extend_towards(
            tree2,
            last1.config,
            distance_fn,
            extend_fn,
            collision_fn,
            not swap,
            **kwargs
        )

        # Drawing only; this does not affect history.
        if draw_fn:
            for sp1, sp2 in zip(tree1, tree2):
                sp1.draw(draw_fn)
                sp2.draw(draw_fn)

        # ---------------------------------------------------------
        # Success:
        # do NOT append another history snapshot.
        # ---------------------------------------------------------
        if success:
            path1, path2 = last1.retrace(), last2.retrace()

            if swap:
                path1, path2 = path2, path1

            if verbose:
                print(
                    'RRT connect: {} iterations, {} nodes'.format(
                        iteration,
                        len(nodes1) + len(nodes2)
                    )
                )

            path = configs(path1[:-1] + path2[::-1])

            return (path, history) if return_history else path

    # -------------------------------------------------------------
    # Failed to find a connection within the iteration/time budget.
    # -------------------------------------------------------------
    return (None, history) if return_history else None


def birrt(start, goal, distance_fn, sample_fn, extend_fn, collision_fn,
          return_history=False, ml_model=None, **kwargs):

    if return_history or ml_model is not None:
        return rrt_connect(
            start,
            goal,
            distance_fn,
            sample_fn,
            extend_fn,
            collision_fn,
            return_history=return_history,
            ml_model=ml_model,
            **kwargs
        )

    from .meta import random_restarts

    solutions = random_restarts(
        rrt_connect,
        start,
        goal,
        distance_fn,
        sample_fn,
        extend_fn,
        collision_fn,
        max_solutions=1,
        **kwargs
    )

    if not solutions:
        return None

    return solutions[0]


#####################################################################


def birrt(start, goal, distance_fn, sample_fn, extend_fn, collision_fn,
          return_history=False, ml_model=None, **kwargs):
    """
    BiRRT entry point.

    When history recording or ML early stopping is requested, directly
    use rrt_connect so that the complete tree history is available.
    Otherwise preserve the original random-restart behavior.
    """

    if return_history or ml_model is not None:
        return rrt_connect(
            start,
            goal,
            distance_fn,
            sample_fn,
            extend_fn,
            collision_fn,
            return_history=return_history,
            ml_model=ml_model,
            **kwargs
        )

    from .meta import random_restarts

    solutions = random_restarts(
        rrt_connect,
        start,
        goal,
        distance_fn,
        sample_fn,
        extend_fn,
        collision_fn,
        max_solutions=1,
        **kwargs
    )

    if not solutions:
        return None

    return solutions[0]