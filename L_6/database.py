from typing import List, Dict, Optional
from L_6.schemas import Person, FaceTemplate


class FaceDatabaseError(Exception):
    pass


class FaceDatabase:
    """
    Mock database for L_6 Face Recognition.
    In a production environment, this would map to PostgreSQL/Milvus.
    """

    def __init__(self):
        self.persons: Dict[str, Person] = {}
        self.templates: Dict[str, FaceTemplate] = {}

    def save_person(self, person: Person) -> None:
        self.persons[person.person_id] = person

    def get_person(self, person_id: str) -> Optional[Person]:
        return self.persons.get(person_id)

    def save_template(self, template: FaceTemplate) -> None:

        MAX_TEMPLATES_PER_PERSON = 5
        existing = [
            t for t in self.templates.values() if t.person_id == template.person_id
        ]
        if len(existing) >= MAX_TEMPLATES_PER_PERSON:
            raise FaceDatabaseError(
                f"Maximum templates ({MAX_TEMPLATES_PER_PERSON}) reached for person {template.person_id}"
            )

        self.templates[template.template_id] = template

    def get_templates_for_person(self, person_id: str) -> List[FaceTemplate]:
        return [t for t in self.templates.values() if t.person_id == person_id]

    def get_all_templates(self) -> List[FaceTemplate]:
        return list(self.templates.values())
