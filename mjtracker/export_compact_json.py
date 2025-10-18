from typing import Dict, List, Any
import pandas as pd
from pathlib import Path
import json
from datetime import datetime


def convert_dataframe_to_compact_json(df: pd.DataFrame, voting_method: str = "approval") -> Dict[str, Any]:
    """
    Convertit le DataFrame des sondages en structure JSON compacte normalisée.

    Args:
        df: DataFrame issu de SurveysInterface après application de la méthode de vote
        voting_method: "approval" ou "majority_judgment"

    Returns:
        Structure JSON optimisée avec métadonnées, candidats et sondages
    """

    # Métadonnées
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "source": "mj2027.csv",
            "voting_method": voting_method,
            "version": "1.0",
        }
    }

    # Types de sondage (normalisé une seule fois)
    poll_types = {}
    poll_type_cols = ["poll_type_id", "institut", "nombre_mentions", "question"]
    mention_cols = [f"mention{i}" for i in range(1, 6)]

    for _, row in df[poll_type_cols + mention_cols].drop_duplicates("poll_type_id").iterrows():
        grades = []
        for i in range(1, int(row["nombre_mentions"]) + 1):
            mention_col = f"mention{i}"
            if mention_col in row and pd.notna(row[mention_col]):
                grades.append({"rank": i, "label": row[mention_col]})

        poll_types[row["poll_type_id"]] = {
            "id": row["poll_type_id"],
            "organization": row["institut"],
            "num_grades": int(row["nombre_mentions"]),
            "question": row["question"],
            "grades": grades,
        }
    output["poll_types"] = poll_types

    # Candidats (normalisé une seule fois)
    candidates = {}
    candidate_cols = ["candidate_id", "name", "surname", "parti"]
    for _, row in df[candidate_cols].drop_duplicates("candidate_id").iterrows():
        candidates[row["candidate_id"]] = {
            "name": f"{row['name']} {row['surname']}",
            "party": row["parti"] if pd.notna(row["parti"]) else "",
            "announced": None,
            "withdrawn": None,
        }
    output["candidates"] = candidates

    # Sondages
    polls = []
    for poll_id in df["poll_id"].unique():
        poll_data = df[df["poll_id"] == poll_id].iloc[0]

        # Résultats par candidat
        results = {}
        for _, row in df[df["poll_id"] == poll_id].iterrows():
            # Distribution des mentions (tableau compact)
            distribution = []
            for i in range(1, int(row["nombre_mentions"]) + 1):
                intent_col = f"intention_mention_{i}"
                if intent_col in row and pd.notna(row[intent_col]):
                    distribution.append(round(float(row[intent_col]), 2))

            candidate_result = {
                "distribution": distribution,
                "no_opinion": round(float(row["sans_opinion"]), 2) if pd.notna(row.get("sans_opinion")) else 0.0,
                "rank": int(row["rang"]),
            }

            # Ajouter les champs spécifiques selon la méthode de vote
            if voting_method == "approval":
                if "approbation" in row and pd.notna(row["approbation"]):
                    candidate_result["approval"] = round(float(row["approbation"]), 2)

            elif voting_method == "majority_judgment":
                # Trouver le rang de la mention majoritaire à partir du label
                if "mention_majoritaire" in row and pd.notna(row["mention_majoritaire"]):
                    mention_maj_label = str(row["mention_majoritaire"])
                    # Chercher le rang correspondant au label
                    for i in range(1, int(row["nombre_mentions"]) + 1):
                        mention_col = f"mention{i}"
                        if (
                            mention_col in row
                            and pd.notna(row[mention_col])
                            and str(row[mention_col]) == mention_maj_label
                        ):
                            candidate_result["median_grade"] = i
                            break

                # Note: il y a une faute de frappe dans les données sources : "majortiaire" au lieu de "majoritaire"
                if "avant_mention_majortiaire" in row and pd.notna(row["avant_mention_majortiaire"]):
                    candidate_result["before_median"] = round(float(row["avant_mention_majortiaire"]), 2)
                elif "avant_mention_majoritaire" in row and pd.notna(row["avant_mention_majoritaire"]):
                    candidate_result["before_median"] = round(float(row["avant_mention_majoritaire"]), 2)

                if "apres_mention_majortiaire" in row and pd.notna(row["apres_mention_majortiaire"]):
                    candidate_result["after_median"] = round(float(row["apres_mention_majortiaire"]), 2)
                elif "apres_mention_majoritaire" in row and pd.notna(row["apres_mention_majoritaire"]):
                    candidate_result["after_median"] = round(float(row["apres_mention_majoritaire"]), 2)

                if "tri_majoritaire" in row and pd.notna(row["tri_majoritaire"]):
                    candidate_result["majority_tie_break"] = str(row["tri_majoritaire"])

            results[row["candidate_id"]] = candidate_result

        poll = {
            "id": poll_id,
            "poll_type_id": poll_data["poll_type_id"],
            "organization": poll_data["institut"],
            "client": poll_data["commanditaire"] if pd.notna(poll_data["commanditaire"]) else "",
            "field_dates": [str(poll_data["start_date"]), str(poll_data["end_date"])],
            "sample_size": int(poll_data["nb_people"]),
            "population": poll_data["population"],
            "results": results,
        }
        polls.append(poll)

    output["polls"] = sorted(polls, key=lambda x: x["field_dates"][1], reverse=True)

    return output


def export_compact_json(df: pd.DataFrame, output_path: Path, voting_method: str = "approval"):
    """
    Exporte le DataFrame en JSON compact.

    Args:
        df: DataFrame des sondages
        output_path: Chemin du fichier de sortie (str ou Path)
        voting_method: "approval" ou "majority_judgment"
    """
    compact_data = convert_dataframe_to_compact_json(df, voting_method)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(compact_data, f, ensure_ascii=False, indent=2)

    print(f"✓ JSON compact exporté ({voting_method}) : {output_path}")
