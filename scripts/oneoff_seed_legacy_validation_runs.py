from __future__ import annotations

from app.db import connect, init_db, seed_legacy_validation_run


def main() -> None:
    init_db()
    with connect() as conn:
        seed_legacy_validation_run(conn)
    print("Seeded legacy validation runs from legacy validation_images / validation_weight_reviews.")


if __name__ == "__main__":
    main()
