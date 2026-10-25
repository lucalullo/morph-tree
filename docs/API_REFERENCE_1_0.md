# Current public API

`from morphtree import MorphTree`

Base use:`MorphTree(random_state=42,verbosity=0).fit(X,y)`.
Task controls:`task="auto"`, `"binary"`, `"multiclass"`, `"regression"`.
Use an explicit task for ambiguous targets. Numeric/categorical columns and
missing values follow the TRAIN-fitted encoder;preserve column names/order.

Implemented controls include random_state,verbosity,time_budget,n_trials,n_jobs,
capacity,max_mutations,use_optuna and existing explicit legacy controls. Current
n_jobs is1;verbosity is0/1;default use_optuna is true;max_mutations defaults2.
No decorative search_budget/max_time/optuna_trials aliases are introduced.

Methods:fit returns self;predict returns labels or values;predict_proba is for
classification. save/load use bounded declarative JSON;load accepts optional
sha256 and explicit workdir. Default load snapshots into an owned temporary
directory without requiring input-directory writes. Wrong hashes,invalid JSON,
inconsistent graphs/recipes and invalid feature widths are rejected.

Inspect with describe_algorithm,get_algorithm_program,get_recipe,
get_mutation_history,get_search_report. Program,recipe,numerical parameters,
learned state and history are distinct. Loaded reports say NOT_REVERIFIED when
their original journal is not reopened. Native learning remains Pure Python.

export_standalone/export_python emit dependency-free inference code.
export_algorithm emits an encoded-training fragment requiring the exact runtime
and admitted context;it is not a standalone trainer or exported full search.
CLI:`python -m morphtree inspect|predict|export --model model.json`.
CLI predict also needs --input;export needs --output. There is no CLI training.
