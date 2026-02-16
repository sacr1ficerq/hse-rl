import torch

device = torch.device("cuda:0")
DEBUG = 0


def conv2d_size_out(size: int, kernel_size: int, stride: int) -> int:
    """
    common use case:
    cur_layer_img_w = conv2d_size_out(cur_layer_img_w, kernel_size, stride)
    cur_layer_img_h = conv2d_size_out(cur_layer_img_h, kernel_size, stride)
    to understand the shape for dense layer's input
    """
    return (size - (kernel_size - 1) - 1) // stride + 1


def compute_td_loss(
    states,
    actions,
    rewards,
    next_states,
    is_done,
    agent,
    target_agent,
    gamma=0.99,
    check_shapes=False,
    device=device,
):
    """Compute td loss using torch operations only"""
    states = torch.tensor(
        states, device=device, dtype=torch.float
    )  # shape: [batch_size, *state_shape]

    # for some torch reason should not make actions a tensor
    actions = torch.tensor(
        actions, device=device, dtype=torch.long
    )  # shape: [batch_size]
    rewards = torch.tensor(
        rewards, device=device, dtype=torch.float
    )  # shape: [batch_size]
    # shape: [batch_size, *state_shape]
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    is_done = torch.tensor(
        is_done.astype("float32"), device=device, dtype=torch.float
    )  # shape: [batch_size]
    is_not_done = 1 - is_done

    # get q-values for all actions in current states
    predicted_qvalues = agent(states)
    assert predicted_qvalues.requires_grad, "qvalues must be a torch tensor with grad"

    # compute q-values for all actions in next states
    with torch.no_grad():
        # <YOUR_CODE>
        predicted_next_qvalues = agent(next_states)  # shape: [batch_size, n_actions]
        target_predicted_next_qvalues = target_agent(
            next_states
        )  # shape: [batch_size, n_actions]

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues.gather(
        dim=1, index=actions.unsqueeze(1)
    ).squeeze()
    if DEBUG:
        print(
            f"predicted_qvalues_for_actions shape: {predicted_qvalues_for_actions.shape}"
        )

    # compute V*(next_states) using predicted next q-values
    max_a_idx = torch.argmax(predicted_next_qvalues, dim=1).unsqueeze(
        1
    )  # shape: [batch_size]
    if DEBUG:
        print(f"max_a shape: {max_a_idx.shape}")
    if DEBUG:
        print(
            f"target_predicted_next_qvalues shape: {target_predicted_next_qvalues.shape}"
        )
    # next_state_values, _ = target_predicted_next_qvalues[:, max_a_idx]
    next_state_values = target_predicted_next_qvalues.gather(
        dim=1, index=max_a_idx
    ).squeeze()  # shape: [batch_size]

    if DEBUG:
        print(f"next_state_values shape: {next_state_values.shape}")

    assert (
        next_state_values.dim() == 1 and next_state_values.shape[0] == states.shape[0]
    ), "must predict one value per state"

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    # you can multiply next state values by is_not_done to achieve this.
    target_qvalues_for_actions = (
        rewards + is_not_done * gamma * next_state_values
    )  # shape: [batch_size]

    assert (
        target_qvalues_for_actions.requires_grad == False
    ), "do not send gradients to target!"

    # mean squared error loss to minimize
    loss = torch.mean(
        (predicted_qvalues_for_actions - target_qvalues_for_actions) ** 2
    )  # order?

    if check_shapes:
        assert (
            predicted_next_qvalues.data.dim() == 2
        ), "make sure you predicted q-values for all actions in next state"
        assert (
            next_state_values.data.dim() == 1
        ), "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        assert (
            target_qvalues_for_actions.data.dim() == 1
        ), "there's something wrong with target q-values, they must be a vector"

    return loss


def play_and_record(initial_state, agent, env, exp_replay, n_steps=1):
    """
    Play the game for exactly n_steps, record every (s,a,r,s', done) to replay buffer.
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has terminated=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time and the state in which the env stays
    """
    s = initial_state
    sum_rewards = 0

    # Play the game for n_steps as per instructions above
    for _ in range(n_steps):
        action = agent.sample_actions([s])[0]
        next_s, r, terminated, truncated, _ = env.step(action)
        exp_replay.add(s, action, r, next_s, terminated)
        sum_rewards += r
        s = next_s
        if terminated or truncated:
            s, _ = env.reset()

    return sum_rewards, s


def play_and_record_prioritized(initial_state, agent, env, exp_replay, n_steps=1):
    """
    Play the game for exactly n_steps, record every (s,a,r,s', done) to replay buffer.
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has terminated=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time and the state in which the env stays
    """
    s = initial_state
    sum_rewards = 0

    # Play the game for n_steps as per instructions above
    for _ in range(n_steps):
        action = agent.sample_actions([s])[0]
        next_s, r, terminated, truncated, _ = env.step(action)
        exp_replay.add(s, action, r, next_s, terminated)
        sum_rewards += r
        s = next_s
        if terminated or truncated:
            s, _ = env.reset()

    return sum_rewards, s


def compute_quantile_loss(
    states,
    actions,
    rewards,
    next_states,
    is_done,
    agent,
    target_agent,
    gamma=0.99,
    device=device,
):
    """Compute quantile loss using torch operations only"""
    states = torch.tensor(
        states, device=device, dtype=torch.float
    )  # shape: [batch_size, *state_shape]

    # for some torch reason should not make actions a tensor
    actions = torch.tensor(
        actions, device=device, dtype=torch.long
    )  # shape: [batch_size]
    rewards = torch.tensor(
        rewards, device=device, dtype=torch.float
    )  # shape: [batch_size]
    # shape: [batch_size, *state_shape]
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    is_done = torch.tensor(
        is_done.astype("float32"), device=device, dtype=torch.float
    )  # shape: [batch_size]
    is_not_done = 1 - is_done

    # get q-values for all actions in current states
    predicted_zvalues = agent(states)  # shape: [batch_size, n_bins, n_actions]
    if DEBUG:
        print(f"predicted_zvalues shape: {predicted_zvalues.shape}")
    assert predicted_zvalues.requires_grad, "zvalues must be a torch tensor with grad"

    # compute z-values for all actions in next states
    with torch.no_grad():
        predicted_next_zvalues = agent(next_states)  # shape: [batch_size, n_bins, n_actions]
        target_predicted_next_zvalues = target_agent(next_states)  # shape: [batch_size, n_bins, n_actions]

    # select q-values for chosen actions
    predicted_zvalues_for_actions = predicted_zvalues.gather(
        dim=2,
        index=actions.view(-1, 1, 1).expand(-1, agent.network.n_bins, 1)
    ).squeeze(2)  # should be [batch_size, n_bins]
    if DEBUG:
        print(f"predicted_zvalues_for_actions shape: {predicted_zvalues_for_actions.shape}")

    # compute y = V*(next_states) using predicted next q-values
    target_predicted_next_qvalues = target_predicted_next_zvalues.mean(1)  # [batch_size, n_actions]
    predicted_next_qvalues = predicted_next_zvalues.mean(1)  # [batch_size, n_actions]

    # a*
    max_a_idx = torch.argmax(predicted_next_qvalues, dim=1).unsqueeze(1)  # shape: [batch_size]
    if DEBUG:
        print(f"max_a shape: {max_a_idx.shape}")
        print(f"target_predicted_next_qvalues shape: {target_predicted_next_qvalues.shape}")

    # Z(s', a*)
    next_state_zvalues = target_predicted_next_zvalues.gather(
        dim=2,
        index=max_a_idx.view(-1, 1, 1).expand(-1, agent.network.n_bins, 1)
    ).squeeze(2)  # shape: [batch_size, n_bins]

    if DEBUG:
        print(f"next_state_zvalues shape: {next_state_zvalues.shape}")

    assert (
        next_state_zvalues.dim() == 2 and next_state_zvalues.shape[0] == states.shape[0]
    ), "must predict multiple bins per state"

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    # you can multiply next state values by is_not_done to achieve this.

    r = rewards.unsqueeze(-1)
    g = is_not_done.unsqueeze(-1) * gamma
    z_next = next_state_zvalues

    # y
    y = r + g * z_next  # shape: [batch_size, n_bins]
    assert (not y.requires_grad), "do not send gradients to target!"

    # wasserstein loss
    N = agent.network.n_bins
    tau = torch.arange(N, dtype=torch.float, device=device) + 1 - 0.5
    assert not tau.requires_grad, "we are not optimizing tau"
    tau_hat = tau / N
    assert not tau_hat.requires_grad, "we are not optimizing tau"

    diff = y.unsqueeze(1) - predicted_zvalues_for_actions.unsqueeze(2)  # shape: [batch_size, n_bins, n_bins]
    tau_weight = tau_hat.view(1, N, 1)

    loss = diff * (tau_weight - (diff < 0).float())  # shape: [batch_size, n_bins, n_bins]
    loss = loss.mean(dim=1).mean(dim=1)  # shape: [batch_size]

    return loss.mean()
