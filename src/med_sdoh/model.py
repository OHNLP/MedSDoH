import csv
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests


class MedTagger:

    MED_TAGGER = "MedTagger-fit-context.jar"

    def __init__(
        self,
        input_dir: str | Path,
        annotation_dir: str | Path,
        model_dir: str | Path,
        schema_path: str | Path | None = None,
    ):

        self.input_dir = Path(input_dir)
        self.annotation_dir = Path(annotation_dir)
        self.model_dir = Path(model_dir)
        self.xml_dir = None

        try:
            self.schema_path = (
                Path(schema_path)
                if schema_path
                else list(self.model_dir.rglob("*.dtd"))[0]
            ).absolute()  # .dtd schema file
        except IndexError:
            raise FileNotFoundError(f"No .dtd schema file found in {self.model_dir}.")

        self.model_name = get_model_name(self.schema_path)

    @property
    def annotation_files(self) -> list[Path]:
        return [
            file_path
            for file_path in self.annotation_dir.rglob("*.ann")
            if file_path.is_file() and not file_path.name.startswith(".")
        ]

    def run_medtagger(self, clean_up_jar: bool = False) -> None:
        """
        Downloads and runs the MedTagger JAR file on the input, annotation, and model directories. Genearate annotation files (.ann) in the annotation directory.

        Args:
            clean_up_jar (bool): Whether to delete the JAR file after execution. Default is False.
        """

        # Check if annotation files already exist
        if self.annotation_files:
            user_input = (
                input(
                    f"Annotation files already exist in {self.annotation_dir}. Running the model will overwrite. Press 'y' to proceed or 'n' to cancel: "
                )
                .strip()
                .lower()
            )
            if user_input not in ["y", "yes"]:
                return

        # URL for the .jar file
        jar_url = f"https://github.com/OHNLP/AgingNLP/releases/download/v0.1.1/{self.MED_TAGGER}"
        jar_path = self.model_dir / self.MED_TAGGER

        # Check if the .jar file exists
        if not jar_path.exists():
            print(f"{jar_path} not found. Downloading...")
            try:
                response = requests.get(jar_url, stream=True)
                response.raise_for_status()  # Raise an exception for HTTP errors
                with open(jar_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
            except requests.RequestException as e:
                print(f"Failed to fetch the .jar file: {e}")
                return

        # Run the .jar file
        try:
            subprocess.run(
                [
                    "java",
                    "-Xms512M",
                    "-Xmx2000M",
                    "-jar",
                    str(jar_path),
                    str(self.input_dir),
                    str(self.annotation_dir),
                    str(self.model_dir),
                ],
                check=True,
            )
            print("MedTagger ran successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to run MedTagger: {e}")
        else:
            if clean_up_jar:
                jar_path.unlink()
                print(f"Deleted {jar_path}.")

    def parse_annotation_file(self, file_path: str) -> list[dict] | None:
        """
        Parses a single .ann file and extracts rows into a list of dictionaries.
        Each dictionary represents a row in the .ann file.
        """
        output_list = []

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Read the .ann file
        with open(file_path, "r") as f:
            ann_output = csv.reader(f, delimiter="\t")
            rows = list(ann_output)

        if not rows:
            return None

        for row in rows:
            # Convert the row into a dictionary
            output_dict = {"file_name": row[0]}
            output_dict.update(
                {
                    element.split("=")[0]
                    .lower(): element.split("=")[1]
                    .strip('"')
                    .lower()
                    for element in row
                    if "=" in element
                }
            )
            output_list.append(output_dict)

        return output_list

    def get_annotation_summary(
        self,
        save_summary: bool = True,
        summary_output_dir: str | Path = ".",
    ) -> pd.DataFrame:
        """
        Writes the annotation results to a summarized CSV file.
        Ensures 'file_name' is the first column and excludes 'start' and 'end' keys.
        """
        # Parse annoatation output files
        annotation_results: list[dict] = []
        for file in (self.annotation_dir).rglob("*.ann"):
            annotation_output = self.parse_annotation_file(file)
            if annotation_output:
                annotation_results.extend(annotation_output)

        # Get unique keys from the list of dictionaries
        column_names = set().union(*(e.keys() for e in annotation_results)) - {
            "start",
            "end",
            "semg",
            "section",
            "sentid",
        }
        column_names = ["file_name"] + sorted(column_names - {"file_name"})
        column_names = [s.capitalize() for s in column_names]

        # Store the output in a dataframe
        df = pd.DataFrame(annotation_results)
        df = df[[col.lower() for col in column_names]]

        if save_summary:
            summary_output_dir = Path(summary_output_dir)
            summary_output_dir.mkdir(parents=True, exist_ok=True)

            summary_output_path = summary_output_dir / "annotation_summary.csv"
            df.to_csv(summary_output_path, index=False)
            print(f"Annotation summary saved to {summary_output_path.absolute()}")

        return df

    def create_combined_xml(self, save_dir=None) -> None:
        """
        Creates a combined XML annotation files for MedTator.
        """
        for annotation_file_path in self.annotation_files:

            input_file_path = (
                self.input_dir / annotation_file_path.with_suffix("").name
            )  # original raw text file
            if not input_file_path.exists():
                raise FileNotFoundError(f"Input file {input_file_path} does not exist.")

            with open(input_file_path, "r") as f:
                raw_notes: str = f.read()  # Original input text

            with open(annotation_file_path, "r") as f:
                annotation_output: str = f.read()

            # Create the root element
            root = ET.Element(self.model_name)

            # Add the TEXT section
            text_element = ET.SubElement(root, "TEXT")
            text_element.text = f"<![CDATA[{raw_notes}]]>"

            # Add the TAGS section
            tags_element = ET.SubElement(root, "TAGS")

            # Parse NLP output and create corresponding XML tags
            for line in annotation_output.strip().splitlines():
                parts = line.split("\t")
                del parts[1]
                tag_attributes = {
                    "spans": "{}~{}".format(
                        parts[3].split("=")[1].strip('"'),
                        parts[4].split("=")[1].strip('"'),
                    ),
                    "id": parts[2].split("=")[1].strip('"')[:2].upper()
                    + str(annotation_output.strip().splitlines().index(line)),
                    "certainty": parts[5].split("=")[1].strip('"').lower(),
                    "status": parts[6].split("=")[1].strip('"').lower(),
                    "experiencer": parts[7].split("=")[1].strip('"').lower(),
                    "text": parts[2].split("=")[1].strip('"'),
                    "exclusion": "",
                    "comment": "",
                }

                # Replace spaces and special characters with underscores
                tag_name = parts[8].split("=")[1].strip('"')
                ET.SubElement(tags_element, tag_name, tag_attributes)

            xml_str = ET.tostring(root, encoding="utf-8", method="xml").decode("utf-8")
            xml_str = xml_str.replace("&lt;![CDATA[", "<![CDATA[").replace(
                "]]&gt;", "]]>"
            )
            xml_str = f'<?xml version="1.0" encoding="UTF-8" ?>\n{xml_str}'

            # By default, save the XML files in the medtator directory
            if not save_dir:
                self.xml_dir = self.input_dir.parent / "medtator"
                self.xml_dir.mkdir(exist_ok=True)
            else:
                self.xml_dir = Path(save_dir)
                self.xml_dir.mkdir(exist_ok=True)
            xml_path = self.xml_dir / f"{annotation_file_path.with_suffix('').name}.xml"
            with open(xml_path, "w") as xml_file:
                xml_file.write(xml_str)
        print(f"XML files saved to {self.xml_dir.absolute()}.")

    def purge_data(
        self, annotation_files=True, input_data=False, xlm_files=False, xml_dir=None
    ) -> None:
        """
        Deletes all files in the annotation directory.
        """
        if annotation_files:
            for file in self.annotation_files:
                file.unlink()
            print(f"Deleted all files in {self.annotation_dir}.")
        if xml_dir is not None:
            self.xml_dir = Path(xml_dir)
        if xlm_files and self.xml_dir:
            for file in self.xml_dir.rglob("*.xml"):
                file.unlink()
            print(f"Deleted all files in {self.xml_dir}.")

        if input_data:
            user_input = input(
                "Are you sure you want to delete all files in the input directory? Press 'y' to proceed or 'n' to cancel: "
            )
            if user_input.strip().lower() in ["y", "yes"]:
                for file in self.input_dir.rglob("*"):
                    file.unlink()
                print(f"Deleted all files in {self.input_dir}.")


def get_model_name(schema_path: str | Path) -> str:
    entity_pattern = r'<!ENTITY\s+name\s+"(.*?)"\s*>'

    with open(schema_path, "r") as file:
        content = file.read()
        match = re.search(entity_pattern, content)
    return match.group(1)
