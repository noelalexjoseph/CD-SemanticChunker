"""
Ground truth test cases for construction drawing RAG evaluation.

Each test case includes:
- question: The query to test
- ground_truth: Expected answer content
- relevant_chunk_types: Expected chunk types that should be retrieved
- keywords: Key terms that should appear in retrieved content
"""

CONSTRUCTION_TEST_CASES = [
    {
        "question": "What is a door schedule?",
        "ground_truth": "A door schedule is a table that lists all doors in a construction project with their specifications including door number, size, type, material, hardware, and fire rating.",
        "relevant_chunk_types": ["table", "text"],
        "keywords": ["door", "schedule", "table", "size", "type"]
    },
    {
        "question": "Where can I find the project information?",
        "ground_truth": "Project information is typically found in the title block, located in the bottom-right corner of the construction drawing sheet. It includes project name, drawing number, revision history, and responsible parties.",
        "relevant_chunk_types": ["title_block", "text"],
        "keywords": ["project", "title", "sheet", "information"]
    },
    {
        "question": "What are general notes in construction drawings?",
        "ground_truth": "General notes are text blocks that provide important construction requirements, code references, material specifications, and installation instructions that apply to the entire project or sheet.",
        "relevant_chunk_types": ["text", "notes"],
        "keywords": ["general", "notes", "requirements", "specifications"]
    },
    {
        "question": "How do I read a floor plan?",
        "ground_truth": "A floor plan is a top-down view of a building showing walls, doors, windows, and room layouts. It includes dimensions, room labels, and references to detail drawings.",
        "relevant_chunk_types": ["viewport", "figure", "text"],
        "keywords": ["floor", "plan", "view", "layout", "room"]
    },
    {
        "question": "What information is in a window schedule?",
        "ground_truth": "A window schedule is a table listing all windows with their mark/ID, size, type, glazing specification, frame material, and performance ratings.",
        "relevant_chunk_types": ["table"],
        "keywords": ["window", "schedule", "table", "glazing", "size"]
    },
    {
        "question": "Where are the revision notes?",
        "ground_truth": "Revision notes are typically found in the title block or a dedicated revision block, showing the revision number, date, description of changes, and who approved them.",
        "relevant_chunk_types": ["title_block", "text"],
        "keywords": ["revision", "changes", "date", "history"]
    },
    {
        "question": "What scale is used in this drawing?",
        "ground_truth": "The drawing scale is indicated in the title block and/or next to each viewport. Common scales include 1:100, 1:50, 1:20 for metric or 1/4\"=1'-0\" for imperial.",
        "relevant_chunk_types": ["title_block", "viewport"],
        "keywords": ["scale", "1:100", "1:50", "drawing"]
    },
    {
        "question": "How do I find material specifications?",
        "ground_truth": "Material specifications are found in general notes, schedules (like finish schedules), and specification sections. They detail material types, grades, and installation requirements.",
        "relevant_chunk_types": ["text", "table", "notes"],
        "keywords": ["material", "specification", "finish", "grade"]
    },
    {
        "question": "What does the reflected ceiling plan show?",
        "ground_truth": "A reflected ceiling plan (RCP) shows the ceiling layout as if viewed from above through a transparent floor. It includes lighting fixtures, ceiling materials, soffits, and MEP elements.",
        "relevant_chunk_types": ["viewport", "text"],
        "keywords": ["ceiling", "reflected", "lighting", "RCP"]
    },
    {
        "question": "Where are the fire-rated assemblies indicated?",
        "ground_truth": "Fire-rated assemblies are indicated on floor plans with specific symbols, in wall type schedules, and in general notes. Fire ratings are typically shown as 1-hour, 2-hour, etc.",
        "relevant_chunk_types": ["text", "table", "notes"],
        "keywords": ["fire", "rated", "hour", "assembly", "wall"]
    }
]
