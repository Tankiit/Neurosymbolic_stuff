from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from .reasoning import Action, RolloutResult


@dataclass
class ActionValueController:
    vectorizer: TfidfVectorizer | None = None
    regressors: Dict[str, Ridge] | None = None
    action_names: Tuple[str, ...] = (Action.CONTINUE.value, Action.REVISE.value, Action.BRANCH.value, Action.STOP.value)

    def fit(self, rollout_rows: Iterable[RolloutResult]) -> "ActionValueController":
        rows = list(rollout_rows)
        if not rows:
            raise ValueError("No rollout data available to fit controller.")

        texts = [row.state for row in rows]
        self.vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
        X = self.vectorizer.fit_transform(texts)
        self.regressors = {}
        for action in self.action_names:
            targets = [1.0 if row.action.value == action else 0.0 for row in rows]
            # reward-shaped supervision: actions with higher reward receive higher target values
            value_targets = [row.reward if row.action.value == action else 0.0 for row in rows]
            model = Ridge(alpha=1.0)
            model.fit(X, value_targets)
            self.regressors[action] = model
        return self

    def predict_action(self, state: str) -> str:
        if self.vectorizer is None or self.regressors is None:
            raise ValueError("Controller has not been fitted yet.")

        X = self.vectorizer.transform([state])
        scores = {}
        for action, model in self.regressors.items():
            scores[action] = float(model.predict(X)[0])
        return max(scores, key=scores.get)

    def predict_scores(self, state: str) -> Dict[str, float]:
        if self.vectorizer is None or self.regressors is None:
            raise ValueError("Controller has not been fitted yet.")

        X = self.vectorizer.transform([state])
        scores = {}
        for action, model in self.regressors.items():
            scores[action] = float(model.predict(X)[0])
        return scores

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "ActionValueController":
        with open(path, "rb") as f:
            return pickle.load(f)
