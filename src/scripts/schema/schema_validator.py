import json
from jsonschema import Draft202012Validator
from pathlib import Path


class TriageSchemaValidator:
    def __init__(self, schema_path: str | Path):
        schema_path = Path(schema_path)
        with schema_path.open("r", encoding='utf-8') as f:
            schema = json.load(f)

        Draft202012Validator.check_schema(schema)

        self.schema = schema
        self.validator = Draft202012Validator(self.schema)

    def validate_instance(self, instance):
        """
        Validate a single instance against a JSON Schema.
        Returns:
          ok: bool
          obj: normalized (or original) instance dict
          errors: list[dict] with {path, message}
        """

        errors = []
        for e in sorted(self.validator.iter_errors(instance), key=lambda e: list(e.path)):
            path = ".".join(str(p) for p in e.path) or "<root>"
            errors.append({"path": path, "message": e.message})

        return (len(errors) == 0), instance, errors

    def print_errors(self, errors):
        for err in errors:
            print(f"- {err['path']}: {err['message']}")

    def validate_many(self, instances):
        results = []
        for i, inst in enumerate(instances):
            ok, obj, errs = self.validate_instance(inst)
            results.append({"idx": i, "ok": ok, "obj": obj, "errors": errs})
        return results
