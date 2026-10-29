"""Reusable algorithm specification, independent of trained tree weights."""
from __future__ import annotations

from dataclasses import dataclass, field
import copy

from .config import AlgorithmGenome, HyperParams
from .persistence import decode_species, read_json, species_payload, write_json
from .public import public_metadata, learning_fingerprint


@dataclass(frozen=True)
class AlgorithmSpecies:
    genome: AlgorithmGenome = field(default_factory=AlgorithmGenome)
    parameters: HyperParams = field(default_factory=HyperParams)
    metadata: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        if not isinstance(self.genome, AlgorithmGenome) or not isinstance(self.parameters, HyperParams):
            raise ValueError("Species needs an AlgorithmGenome and HyperParams")
        if not isinstance(self.metadata, dict):
            raise ValueError("Species metadata must be a dict")
        object.__setattr__(self, "metadata", public_metadata(self.metadata))

    @property
    def learning_fingerprint(self):
        recipe = self.metadata.get('learning_recipe')
        return learning_fingerprint(self.genome, self.parameters, recipe) if recipe else None

    @property
    def structural_fingerprint(self):
        return self.genome.fingerprint()

    def to_dict(self):
        return species_payload(self.genome, self.parameters, copy.deepcopy(self.metadata))

    def save(self, path):
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path, *, limits=None):
        genome, params, metadata = decode_species(read_json(path, limits))
        return cls(genome, params, metadata)

    def export_python(self, path):
        from .exporting import export_algorithm
        export_algorithm(self, path)
