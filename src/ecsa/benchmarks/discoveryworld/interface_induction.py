"""Evidence-driven semantic contracts over public, opaque action transitions.

DiscoveryWorld defines a JSON *wire* protocol. This module does not assign
scientific roles to field names, action names or objects: role bindings and
predicted effects come from observed transitions and remain hypotheses until
validated on disjoint evidence. Evaluator-only fields are forbidden.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from hashlib import blake2b
from math import asinh, isfinite, sqrt
from pathlib import Path
from typing import Any
import json

import numpy as np

from .jepa_replay import _jsonl


_FORBIDDEN = frozenset({
    'scoringinfo', 'criticalquestions', 'criticalhypotheses', 'scorecard',
    'finalscorecard', 'resonancefreq', 'isactivated', 'worldhistory',
    'exportworldhistoryjson',
})
_TRANSIENT = frozenset({'worldstep', 'step'})


def _check_public(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError('public JSON keys must be strings')
            if key.lower().replace('_', '') in _FORBIDDEN:
                raise ValueError(f'oracle field in observation/action: {key}')
            _check_public(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_public(item)
    elif value is not None and type(value) not in (bool, int, float, str):
        raise ValueError('unsupported public JSON scalar')
    elif type(value) is float and not isfinite(value):
        raise ValueError('nonfinite public value')


@dataclass(frozen=True)
class PublicTransition:
    evidence_id: str
    before: dict
    action: dict
    after: dict
    success: bool | None

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_id, str) or not self.evidence_id:
            raise ValueError('unique nonempty evidence ID required')
        if not all(isinstance(v, dict) for v in (self.before, self.action, self.after)):
            raise TypeError('public observations and action must be objects')
        if self.success is not None and type(self.success) is not bool:
            raise TypeError('public success must be bool or unknown')
        for payload in (self.before, self.action, self.after):
            _check_public(payload)
        if not isinstance(self.action.get('action'), str) and type(self.action.get('chosen_dialog_option_int')) is not int:
            raise ValueError('action packet requires public action ID')


def _action_schema(packet: dict) -> str:
    return str(packet['action']) if isinstance(packet.get('action'), str) else 'DIALOG_OPTION'


def _arguments(packet: dict) -> tuple[tuple[str, Any], ...]:
    # Names/arity are wire observations. Object roles are NOT assigned by a map.
    return tuple((name, value) for name, value in sorted(packet.items())
                 if name not in ('action', 'chosen_dialog_option_int'))


def _entities(value: Any) -> dict[str, dict[str, object]]:
    """Use public transport identity only for association, never as a feature."""
    seen: dict[str, dict[str, object]] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            ref = node.get('uuid')
            if type(ref) in (str, int) and ref != '':
                fields = {}
                for k, v in node.items():
                    if k == 'uuid' or k.lower().replace('_','') in _TRANSIENT:
                        continue
                    if v is None or type(v) in (bool, int, float, str):
                        fields[k] = v
                key=str(ref)
                if key in seen:
                    for feature,value in fields.items():
                        if feature in seen[key] and seen[key][feature]!=value:
                            raise ValueError('conflicting public records for one source identity')
                    seen[key].update(fields)
                else:
                    seen[key]=fields
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node: walk(child)
    walk(value)
    return seen


def _global_scalars(value:Any)->dict[str,object]:
    """Transport-agnostic scalar paths outside identified object records."""
    values:dict[str,list[object]]=defaultdict(list)
    def walk(item:Any,path:str)->None:
        if isinstance(item,dict):
            # Object features use identity-aligned projection instead.
            if type(item.get('uuid')) in (str,int):return
            for key,part in sorted(item.items()):
                if key.lower().replace('_','') in _TRANSIENT:continue
                walk(part, f'{path}/{key}' if path else key)
        elif isinstance(item,list):
            for part in item:walk(part, path+'[]')
        elif item is None or type(item) in (bool,int,float,str):
            values[path].append(item)
    walk(value,'')
    # Unidentified arrays expose only a multiset, never a trustworthy order.
    return {path:items[0] if len(items)==1 else tuple(sorted(items,key=repr))
            for path,items in values.items()}


def _kind(before: object, after: object) -> str:
    if type(before) in (int, float) and type(after) in (int,float):
        return 'numeric_change'
    if type(before) is bool and type(after) is bool:
        return 'boolean_change'
    return 'categorical_change'


@dataclass(frozen=True, order=True)
class EffectKey:
    role: str
    feature: str
    kind: str


@dataclass(frozen=True)
class BoundRole:
    parameter_name: str
    object_reference_fraction: float
    observations: int


@dataclass(frozen=True)
class EffectHypothesis:
    role: str
    feature: str
    kind: str
    support: int
    total_examples: int
    confidence: float
    supporting_evidence_ids: tuple[str, ...]

    @property
    def key(self) -> EffectKey:
        return EffectKey(self.role, self.feature, self.kind)


@dataclass(frozen=True)
class PreconditionHypothesis:
    role: str
    feature: str
    value: bool | str
    supporting_evidence_ids: tuple[str,...]
    contradicting_evidence_ids: tuple[str,...]
    confidence: float


class InterfaceStatus(StrEnum):
    PROPOSED = 'proposed'
    SUPPORTED = 'supported'
    # ADMITTED is deliberately unattainable through observational replay:
    # intervention/repeatability evidence needs a separate live protocol.
    ADMITTED = 'admitted'
    REJECTED = 'rejected'
    INSUFFICIENT = 'insufficient'


@dataclass(frozen=True)
class ActionContract:
    schema_id: str
    parameter_names: tuple[str, ...]
    roles: tuple[BoundRole, ...]
    effects: tuple[EffectHypothesis, ...]
    preconditions: tuple[PreconditionHypothesis,...]
    status: InterfaceStatus
    success_rate: float | None
    training_evidence_ids: tuple[str, ...]
    observed_properties: tuple[str, ...]


@dataclass(frozen=True)
class ContractValidation:
    schema_id: str
    status: InterfaceStatus
    correct_effect_rate: float
    train_evidence_count: int
    validation_evidence_count: int
    false_effect_count: int
    effect_precision: float
    effect_recall: float
    effect_f1: float
    causally_validated: bool = False
    applicability_accuracy: float | None = None


@dataclass(frozen=True)
class ContractEvaluation:
    evaluated_count: int
    correct_effect_rate: float
    per_schema: tuple[ContractValidation, ...]


@dataclass(frozen=True)
class InterfaceRevision:
    previous: tuple[ActionContract,...]
    current: tuple[ActionContract,...]
    requires_independent_validation: bool


def _observed_effects(item: PublicTransition) -> set[EffectKey]:
    old=_entities(item.before)
    new=_entities(item.after)
    arg_binding={str(v): k for k,v in _arguments(item.action) if type(v) in (str,int) and str(v) in old}
    found: set[EffectKey] = set()
    for ref in old.keys() & new.keys():
        left,right=old[ref],new[ref]
        role=arg_binding.get(ref,'unbound')
        for key in left.keys() & right.keys():
            if left[key] != right[key]:
                found.add(EffectKey(role,key,_kind(left[key],right[key])))
    g_before=_global_scalars(item.before)
    g_after=_global_scalars(item.after)
    for key in g_before.keys() & g_after.keys():
        if g_before[key]!=g_after[key]:
            found.add(EffectKey('global',key,_kind(g_before[key],g_after[key])))
    return found


def _observed_conditions(row:PublicTransition)->dict[tuple[str,str],bool|str]:
    """Bound pre-action categorical attributes. Numeric thresholds require a separate learner."""
    source=_entities(row.before)
    conditions={}
    for role,value in _arguments(row.action):
        if type(value) not in (str,int) or str(value) not in source:continue
        for feature,content in source[str(value)].items():
            if type(content) in (bool,str) and (type(content) is bool or len(content)<=48):
                conditions[(role,feature)]=content
    return conditions


def _propose_preconditions(rows:list[PublicTransition],min_support:int)->tuple[PreconditionHypothesis,...]:
    known=[r for r in rows if r.success is not None]
    positives=sum(r.success for r in known)
    negatives=len(known)-positives
    if positives<min_support or negatives<min_support:return ()
    mapped=[_observed_conditions(r) for r in known]
    variants={ (role,feature,value) for row in mapped for (role,feature),value in row.items() }
    proposals=[]
    for role,feature,value in sorted(variants,key=repr):
        present=[i for i,v in enumerate(mapped) if v.get((role,feature))==value]
        absent=[i for i in range(len(known)) if i not in present]
        if len(present)<min_support or len(absent)<min_support:continue
        yes=sum(bool(known[i].success) for i in present)
        no=sum(bool(known[i].success) for i in absent)
        # Smoothed likelihood separation: avoids a perfect rule from one trial.
        p_yes=(yes+1)/(len(present)+2)
        p_no=(no+1)/(len(absent)+2)
        if p_yes-p_no<.5:continue
        proposals.append(PreconditionHypothesis(
            role,feature,value,
            tuple(known[i].evidence_id for i in present if known[i].success),
            tuple(known[i].evidence_id for i in present if not known[i].success),
            p_yes-p_no,
        ))
    return tuple(proposals)


class InterfaceLearner:
    """Induce candidate action interface and effects; validate without leakage."""

    def __init__(self, *, min_support: int = 3, min_consistency: float = .6,
                 max_recent_examples: int | None = None):
        if type(min_support) is not int or min_support < 1:
            raise ValueError('min_support must be positive')
        if not isfinite(min_consistency) or not 0 < min_consistency <= 1:
            raise ValueError('min_consistency must be in (0,1]')
        if max_recent_examples is not None and (
                type(max_recent_examples) is not int or max_recent_examples < min_support):
            raise ValueError('max_recent_examples must cover minimum evidence')
        self.min_support=min_support
        self.min_consistency=min_consistency
        self.max_recent_examples=max_recent_examples
        self._rows: list[PublicTransition]=[]
        self._ids: set[str]=set()

    def observe(self, transition: PublicTransition) -> None:
        if not isinstance(transition,PublicTransition):
            raise TypeError('public transition required')
        if transition.evidence_id in self._ids:
            raise ValueError('duplicate training evidence')
        self._rows.append(transition)
        self._ids.add(transition.evidence_id)
        if self.max_recent_examples is not None:
            same=[row for row in self._rows if _action_schema(row.action)==_action_schema(transition.action)]
            if len(same)>self.max_recent_examples:
                self._rows.remove(same[0])

    def adapt(self, transitions:tuple[PublicTransition,...])->InterfaceRevision:
        """Propose a new interface revision from later observations, not oracle labels.

        Existing snapshots stay immutable; new candidates require separate
        validation before they can be trusted for planning.
        """
        if not isinstance(transitions,tuple) or not transitions:
            raise ValueError('adaptation requires nonempty immutable evidence')
        before=self.proposals()
        for row in transitions:self.observe(row)
        return InterfaceRevision(before,self.proposals(),True)

    def proposals(self) -> tuple[ActionContract,...]:
        grouped: dict[str,list[PublicTransition]]=defaultdict(list)
        for row in self._rows: grouped[_action_schema(row.action)].append(row)
        proposals=[]
        for schema, rows in sorted(grouped.items()):
            parameters=tuple(sorted({k for row in rows for k,_ in _arguments(row.action)}))
            roles=[]
            for param in parameters:
                candidate=[(row,dict(_arguments(row.action))[param]) for row in rows if param in dict(_arguments(row.action))]
                count=sum(type(value) in (str,int) and str(value) in _entities(row.before)
                          for row,value in candidate)
                roles.append(BoundRole(param,count/max(1,len(candidate)),len(candidate)))
            evidence: dict[EffectKey,list[str]]=defaultdict(list)
            properties: set[str]=set()
            for row in rows:
                for values in _entities(row.before).values():properties.update(values)
                properties.update('global:'+key for key in _global_scalars(row.before))
                for effect in _observed_effects(row):evidence[effect].append(row.evidence_id)
            effects=tuple(EffectHypothesis(effect.role,effect.feature,effect.kind,len(ids),len(rows),
                                           len(ids)/len(rows),tuple(ids))
                          for effect,ids in sorted(evidence.items()) if
                          len(ids)>=self.min_support and len(ids)/len(rows)>=self.min_consistency)
            outcomes=[row.success for row in rows if row.success is not None]
            proposals.append(ActionContract(
                schema_id=schema,parameter_names=parameters,roles=tuple(roles),effects=effects,
                preconditions=_propose_preconditions(rows,self.min_support),
                status=InterfaceStatus.PROPOSED,
                success_rate=(sum(outcomes)/len(outcomes) if outcomes else None),
                training_evidence_ids=tuple(row.evidence_id for row in rows),
                observed_properties=tuple(sorted(properties)),
            ))
        return tuple(proposals)

    def _audit(self, rows: tuple[PublicTransition,...], *, min_validation: int) -> ContractEvaluation:
        if any(row.evidence_id in self._ids for row in rows):
            raise ValueError('validation/training evidence overlap')
        if len({row.evidence_id for row in rows}) != len(rows):
            raise ValueError('duplicate validation evidence')
        per=[]
        by_schema: dict[str,list[PublicTransition]]=defaultdict(list)
        for row in rows:by_schema[_action_schema(row.action)].append(row)
        for contract in self.proposals():
            cohort=by_schema[contract.schema_id]
            if not cohort:continue
            expected={effect.key for effect in contract.effects}
            matched=0
            false_effects=0
            true_positive=0
            false_negative=0
            for row in cohort:
                actual=_observed_effects(row)
                matched+=int(bool(expected) and expected.issubset(actual))
                false_effects+=len(expected-actual)
                true_positive+=len(expected&actual)
                false_negative+=len(actual-expected)
            rate=matched/len(cohort)
            precision=true_positive/max(1,true_positive+false_effects)
            recall=true_positive/max(1,true_positive+false_negative)
            f1=(2*precision*recall/(precision+recall) if precision+recall else 0.)
            applicability=None
            labeled=[row for row in cohort if row.success is not None]
            if contract.preconditions and labeled:
                applicability=sum(
                    (all(_observed_conditions(row).get((p.role,p.feature))==p.value
                         for p in contract.preconditions)==row.success)
                    for row in labeled
                )/len(labeled)
            supported=((bool(expected) and rate>=self.min_consistency and f1>=self.min_consistency)
                       or (applicability is not None and applicability>=self.min_consistency))
            status=(InterfaceStatus.INSUFFICIENT if len(cohort)<min_validation or (not expected and applicability is None)
                    else InterfaceStatus.SUPPORTED if supported
                    else InterfaceStatus.REJECTED)
            per.append(ContractValidation(contract.schema_id,status,rate,
                           len(contract.training_evidence_ids),len(cohort),false_effects,
                           precision,recall,f1,applicability_accuracy=applicability))
        total=sum(p.validation_evidence_count for p in per)
        weighted=sum(p.correct_effect_rate*p.validation_evidence_count for p in per)
        return ContractEvaluation(total,weighted/max(1,total),tuple(per))

    def validate(self, rows: tuple[PublicTransition,...], *,min_validation:int=4) -> tuple[ContractValidation,...]:
        if type(min_validation) is not int or min_validation<1:
            raise ValueError('min_validation must be positive')
        return self._audit(rows,min_validation=min_validation).per_schema

    def evaluate(self, rows: tuple[PublicTransition,...]) -> ContractEvaluation:
        return self._audit(rows,min_validation=1)


class LearnedInterfaceEncoder:
    """Frozen, role-relative projection from *learned* interface vocabulary.

    Public identity is used for binding within a single observation, but never
    hashed as a predictive feature. There are no domain-specific effect names.
    """

    def __init__(self, vocabulary: tuple[str,...], *, width:int, action_width:int):
        self.vocabulary=vocabulary
        self.width=width
        self.action_width=action_width

    @classmethod
    def fit(cls, contracts:tuple[ActionContract,...],*,width:int=256,
            action_width:int|None=None)->'LearnedInterfaceEncoder':
        if type(width) is not int or width<8:
            raise ValueError('feature width must be >=8')
        if action_width is None: action_width=max(8,width//2)
        if type(action_width) is not int or action_width<8:
            raise ValueError('action width must be >=8')
        if not isinstance(contracts,tuple) or not all(isinstance(c,ActionContract) for c in contracts):
            raise TypeError('typed proposed contracts required')
        features=set()
        for contract in contracts:
            features.update(contract.observed_properties)
            features.update(e.feature for e in contract.effects)
        return cls(tuple(sorted(features)),width=width,action_width=action_width)

    def encode_action(self, packet:dict, observation:dict)->np.ndarray:
        """Infer argument roles from public identity matches, not argument names.

        Integer object IDs are never used as predictive numeric targets.
        """
        _check_public(packet);_check_public(observation)
        entities=_entities(observation)
        vec=np.zeros(self.action_width,dtype=np.float32)

        def push(label:str,value:float=1.) -> None:
            h=blake2b(label.encode(),digest_size=8,person=b'ecsa-if2').digest()
            idx=int.from_bytes(h[:4],'little')%self.action_width
            vec[idx]+=(1. if h[4]&1 else -1.)*value

        push('action:'+_action_schema(packet))
        for name,value in _arguments(packet):
            if type(value) in (str,int) and str(value) in entities:
                push('bound:'+name)
                for feature in self.vocabulary:
                    if feature not in entities[str(value)]:continue
                    v=entities[str(value)][feature]
                    if type(v) in (int,float):
                        push('bound-value:'+name+':'+feature,float(np.clip(asinh(float(v))/3.,-3,3)))
                    else:
                        push('bound-feature:'+name+':'+feature)
            else:
                kind='numeric' if type(value) in (int,float) else 'boolean' if type(value) is bool else 'categorical'
                push('literal:'+name+':'+kind)
        return vec

    def encode_observation(self, observation:dict)->np.ndarray:
        """Action-independent E(o): the JEPA target must not see the action."""
        return self._encode_state(observation,{})

    def encode_before(self, observation:dict, action:dict)->np.ndarray:
        """Optional role-bound diagnostic view; NOT the JEPA target encoder."""
        _check_public(action)
        source=_entities(observation)
        bound={str(value):name for name,value in _arguments(action)
               if type(value) in (str,int) and str(value) in source}
        return self._encode_state(observation,bound)

    def _encode_state(self, observation:dict, bound:dict[str,str])->np.ndarray:
        _check_public(observation)
        entities=_entities(observation)
        vector=np.zeros(self.width,dtype=np.float32)
        vocabulary=set(self.vocabulary)
        def add(token:str,value:object)->None:
            digest=blake2b(token.encode(),digest_size=8,person=b'ecsa-if1').digest()
            idx=int.from_bytes(digest[:4],'little')%self.width
            sign=1 if digest[4]&1 else -1
            if type(value) in (int,float):
                weight=float(np.clip(asinh(float(value))/3.,-3,3))
            elif type(value) is bool:
                weight=1. if value else -1.
            else:
                text=str(value).lower() if value is not None else '<null>'
                category=blake2b(text.encode(),digest_size=8,person=b'ecsa-val').digest()
                idx=(idx+int.from_bytes(category[:4],'little'))%self.width
                weight=1.
            vector[idx]+=sign*weight
        for ref, features in entities.items():
            role=bound.get(ref,'unbound')
            for key,value in features.items():
                if key not in vocabulary:continue
                add(f'{role}:{key}',value)
        if len(entities):vector/=sqrt(len(entities))
        for key,value in _global_scalars(observation).items():
            if 'global:'+key in vocabulary:
                add('global:'+key,value)
        return vector


def load_contract_episode(directory:Path)->tuple[PublicTransition,...]:
    """Read public Arena A logs. Scorecards and oracle history are never opened."""
    directory=Path(directory)
    metadata=json.loads((directory/'run.json').read_text(encoding='utf-8'))
    if (not isinstance(metadata,dict) or type(metadata.get('seed')) is not int
        or metadata.get('scenario')!='Reactor Lab' or metadata.get('difficulty')!='Normal'):
        raise ValueError('identified Reactor Lab / Normal public run metadata required')
    actions=_jsonl(directory/'actions.jsonl')
    observations=_jsonl(directory/'observations.jsonl')
    outcome_path=directory/'action_outcomes.jsonl'
    outcomes=_jsonl(outcome_path) if outcome_path.exists() else []
    results:dict[int,bool]={}
    for row in outcomes:
        if type(row.get('step')) is int and type(row.get('success')) is bool:
            if row['step']-1 in results:raise ValueError('duplicate public action outcome')
            results[row['step']-1]=row['success']
    lookup={}
    for row in observations:
        key=(row.get('step'),row.get('phase'))
        if type(key[0]) is not int or key[1] not in ('pre','post') or not isinstance(row.get('observation'),dict):
            raise ValueError('malformed public observation entry')
        if key in lookup:raise ValueError('duplicate public observation step/phase')
        lookup[key]=row['observation']
    records=[]
    seen_actions=set()
    for row in actions:
        step=row.get('step')
        if type(step) is not int or step<0:
            raise ValueError('invalid action step')
        if step in seen_actions:raise ValueError('duplicate public action step')
        seen_actions.add(step)
        if (step,'pre') not in lookup or (step+1,'post') not in lookup:
            raise ValueError('public action/observation alignment failure')
        records.append(PublicTransition(
            evidence_id=f'dw:seed-{metadata["seed"]}:{directory.name}:{step}',
            before=lookup[(step,'pre')],action=row['action'],after=lookup[(step+1,'post')],
            success=results.get(step),
        ))
    return tuple(records)


@dataclass(frozen=True)
class InductionSplits:
    splits: object
    contracts: tuple[ActionContract,...]
    contract_validation: ContractEvaluation
    contract_heldout: ContractEvaluation
    feature_vocabulary: tuple[str,...]
    unseen_test_properties: tuple[str,...]


def load_interface_splits(root:Path, *,arm:str='cold',mode:str='study',
                          observation_dim:int=256,action_dim:int=128,
                          min_support:int=3)->InductionSplits:
    """Build independent train/val/test vectors using only TRAIN-discovered contracts."""
    from .jepa_replay import ReplaySplits, ReplayTransitions
    if mode not in ('study','smoke') or arm not in ('cold','reuse'):
        raise ValueError('supported mode=study/smoke and arm=cold/reuse')
    root=Path(root)
    folders=sorted(root.glob(f'seed-*/{arm}'),key=lambda p:int(p.parent.name[5:]))
    if not folders and root.name==arm and (root/'actions.jsonl').exists():
        folders=[root]
    if not folders:
        raise ValueError('no public episode logs found')
    episodes={}
    for path in folders:
        metadata=json.loads((path/'run.json').read_text(encoding='utf-8'))
        seed=metadata.get('seed')
        if type(seed) is not int or seed in episodes:
            raise ValueError('unique numeric seed required')
        episodes[seed]=load_contract_episode(path)
    seeds=tuple(sorted(episodes))
    if mode=='study':
        if len(seeds)<3:raise ValueError('study requires three independent seeds')
        train_seeds,val_seeds,test_seeds=seeds[:-2],(seeds[-2],),(seeds[-1],)
        train_rows=tuple(r for s in train_seeds for r in episodes[s])
        val_rows=episodes[val_seeds[0]]
        test_rows=episodes[test_seeds[0]]
        split_kind='independent_seed_holdout'
    else:
        if len(seeds)!=1:raise ValueError('smoke requires one logged seed')
        rows=episodes[seeds[0]];n=len(rows)
        if n<15:raise ValueError('smoke requires at least 15 public transitions')
        train_rows,val_rows,test_rows=(rows[:int(n*.6)],
                                      rows[int(n*.6):int(n*.8)],rows[int(n*.8):])
        train_seeds=val_seeds=test_seeds=seeds
        split_kind='single_episode_temporal_probe_only'
    learner=InterfaceLearner(min_support=min_support)
    for item in train_rows:learner.observe(item)
    contracts=learner.proposals()
    encoder=LearnedInterfaceEncoder.fit(contracts,width=observation_dim,action_width=action_dim)
    test_features={k for row in test_rows for entity in _entities(row.before).values() for k in entity}
    test_features.update('global:'+k for row in test_rows for k in _global_scalars(row.before))

    def matrix(rows:tuple[PublicTransition,...])->ReplayTransitions:
        return ReplayTransitions(
            np.asarray([encoder.encode_observation(r.before) for r in rows],dtype=np.float32),
            np.asarray([encoder.encode_action(r.action,r.before) for r in rows],dtype=np.float32),
            np.asarray([encoder.encode_observation(r.after) for r in rows],dtype=np.float32),
            tuple(_action_schema(r.action) for r in rows), tuple(range(len(rows))),
        )
    return InductionSplits(
        splits=ReplaySplits(matrix(train_rows),matrix(val_rows),matrix(test_rows),
                           train_seeds,val_seeds,test_seeds,split_kind),
        contracts=contracts,
        contract_validation=learner.evaluate(val_rows),
        contract_heldout=learner.evaluate(test_rows),
        feature_vocabulary=encoder.vocabulary,
        unseen_test_properties=tuple(sorted(test_features-set(encoder.vocabulary))),
    )
