"""Preview in-memory training/evaluation dataset exclusions without rewriting folds."""

if __package__:
    from .decoder_va.filters import main
else:
    from decoder_va.filters import main


if __name__ == "__main__":
    raise SystemExit(main())
