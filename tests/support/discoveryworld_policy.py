from ecsa.benchmarks.discoveryworld.contracts import PolicyDecision


class DeterministicPolicy:
    def __init__(self, config):
        self.config = dict(config)

    def decide(
        self,
        observation,
        available_actions,
        teleport_locations,
        scientific_context,
    ):
        return PolicyDecision(
            action={"action": "DISCOVERY_FEED_GET_UPDATES"},
            memory="deterministic-test",
        )


def create_policy(config):
    return DeterministicPolicy(config)


def create_not_policy(config):
    return object()


not_callable = 42
