from __future__ import annotations

from copy import deepcopy

from .contracts import (
    ActionPacket,
    DiscoveryWorldActionResult,
    DiscoveryWorldEpisodeConfig,
    DiscoveryWorldEvaluation,
    JSONValue,
    validate_action_packet,
)

DISCOVERYWORLD_REPOSITORY = "https://github.com/allenai/discoveryworld.git"
DISCOVERYWORLD_REVISION = "fd591323920be0d3786ef350955de1945aa571e5"


try:
    from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "DiscoveryWorld integration requires the optional 'discoveryworld' "
        "dependency: pip install -e '.[discoveryworld]'"
    ) from exc


class DiscoveryWorldEnvironmentAdapter:
    """Strict policy-facing wrapper around the official DiscoveryWorld API."""

    def __init__(
        self,
        api: DiscoveryWorldAPI,
        config: DiscoveryWorldEpisodeConfig,
    ) -> None:
        self._api = api
        self.config = config
        self._steps = 0

    @classmethod
    def reactor_lab_normal(
        cls,
        seed: int,
        *,
        max_steps: int = 1000,
        thread_id: int = 0,
    ) -> "DiscoveryWorldEnvironmentAdapter":
        if type(seed) is not int or seed not in range(5):
            raise ValueError(
                "Reactor Lab benchmark seed must be one of 0, 1, 2, 3, 4"
            )
        config = DiscoveryWorldEpisodeConfig(
            scenario="Reactor Lab",
            difficulty="Normal",
            seed=seed,
            max_steps=max_steps,
        )
        api = DiscoveryWorldAPI(threadID=thread_id)
        loaded = api.loadScenario(
            scenarioName=config.scenario,
            difficultyStr=config.difficulty,
            randomSeed=config.seed,
            numUserAgents=1,
        )
        if not loaded:
            raise RuntimeError(
                "DiscoveryWorld failed to load Reactor Lab / Normal"
            )
        return cls(api, config)

    def observe(self) -> dict[str, JSONValue]:
        observation = self._api.getAgentObservation(agentIdx=0)
        if not isinstance(observation, dict):
            raise RuntimeError("DiscoveryWorld returned a non-dict observation")
        # Isolate policy data from later mutations in the upstream UI/world.
        return deepcopy(observation)

    def available_actions(self) -> dict[str, JSONValue]:
        actions = self._api.listKnownActions(limited=False)
        if not isinstance(actions, dict):
            raise RuntimeError("DiscoveryWorld returned invalid action metadata")
        return deepcopy(actions)

    def teleport_locations(self) -> dict[str, JSONValue]:
        locations = self._api.listTeleportLocationsDict()
        if not isinstance(locations, dict):
            raise RuntimeError(
                "DiscoveryWorld returned invalid teleport metadata"
            )
        return deepcopy(locations)

    def act(self, action: ActionPacket) -> DiscoveryWorldActionResult:
        if self._steps >= self.config.max_steps:
            raise RuntimeError("DiscoveryWorld episode step budget exhausted")
        validate_action_packet(action)

        response = None
        try:
            response = self._api.performAgentAction(
                agentIdx=0,
                actionJSON=dict(action),
            )
        finally:
            tick_result = self._api.tick()
            if not isinstance(tick_result, dict) or not tick_result.get(
                "success", False
            ):
                raise RuntimeError("DiscoveryWorld tick failed")
            self._steps += 1

        if not isinstance(response, dict):
            raise RuntimeError("DiscoveryWorld returned an invalid action result")
        success = response.get("success")
        if type(success) is not bool:
            raise RuntimeError("DiscoveryWorld action result has no bool success")
        errors_raw = response.get("errors", ())
        # The pinned upstream returns an empty string for dialog actions
        # while ordinary actions use a list. Normalize that API quirk here.
        if errors_raw is None or errors_raw == "":
            errors = ()
        elif isinstance(errors_raw, str):
            errors = (errors_raw,)
        elif isinstance(errors_raw, (list, tuple)):
            errors = tuple(str(error) for error in errors_raw)
        else:
            raise RuntimeError(
                "DiscoveryWorld action errors must be a sequence or string"
            )
        return DiscoveryWorldActionResult(
            success=success,
            errors=errors,
        )

    @property
    def done(self) -> bool:
        return bool(self._api.areTasksComplete())

    @property
    def steps(self) -> int:
        # Upstream getStepCounter() is not incremented at the pinned revision.
        # Count the exact performAgentAction -> tick interactions here instead.
        return self._steps

    def evaluate_after_run(self) -> DiscoveryWorldEvaluation:
        scorecard = self._api.getTaskScorecard()
        if not isinstance(scorecard, list) or not scorecard:
            raise RuntimeError("DiscoveryWorld returned no task scorecard")
        first = scorecard[0]
        if not isinstance(first, dict):
            raise RuntimeError("DiscoveryWorld task scorecard is malformed")
        completed = first.get("completedSuccessfully")
        score = first.get("scoreNormalized")
        if type(completed) is not bool:
            raise RuntimeError(
                "DiscoveryWorld scorecard has invalid completedSuccessfully"
            )
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise RuntimeError(
                "DiscoveryWorld scorecard has invalid scoreNormalized"
            )
        return DiscoveryWorldEvaluation(
            completed_successfully=completed,
            score_normalized=float(score),
            steps=self._steps,
            scorecard=deepcopy(scorecard),
        )
