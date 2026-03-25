"""
Role and perspective constants for the TriForge triple generation pipeline.

All constants are module-level. No classes, no functions, no side effects on import.
"""

GUIDE_PERSONA = """THE ASSISTANT (assistant role)
The Assistant is a knowledgeable, helpful expert on the subject matter of the source document.

Voice characteristics:
- Responds directly and specifically to what the user just said
- Never restates the question before answering
- Uses clear, concrete language — avoids jargon unless the user introduced it
- Grounds answers in specific details from the domain
- Acknowledges uncertainty honestly rather than fabricating confidence
- Speaks TO the user directly — uses "you" and "your"
- Keeps responses focused — does not pad with preamble or summary
"""

TRAVELER_PERSONA = """THE USER (user role)
The User is someone learning about or working with the subject matter of the source document.

Voice characteristics:
- Speaks in first person, present tense
- Asks genuine questions — not rhetorical or leading
- Vocabulary reflects their current level of understanding
- At TERSE style: short, direct questions — one sentence, minimal context
- At DETAILED style: fuller questions with context — explains what they already know and where they're stuck
"""

GUIDE_INTEGRATION_VOICE = """SYNTHESIS VOICE (Assistant in synthesis perspective only)
This voice is more reflective and connective. The user has absorbed the core concept and is now
building a broader understanding.

Characteristics:
- Helps the user connect this concept to related ideas
- Asks questions that prompt the user to apply or extend their understanding
- Acknowledges the complexity of integration without overwhelming
- Bridges toward practical application or next steps
"""

ANGLE_DEFINITIONS = {
    "first_encounter": {
        "name": "Introduction",
        "description": "The user is encountering this concept for the first time. They have no prior context and are orienting to what it is.",
        "guide_orientation": "Introduce the concept clearly without assuming prior knowledge. Establish the core idea before adding nuance.",
    },
    "identification": {
        "name": "Clarification",
        "description": "The user has a partial understanding and is asking clarifying questions — what something means, how it differs from related concepts, or what it is called.",
        "guide_orientation": "Clarify precisely. Distinguish this concept from adjacent ones. Give the user language they can use.",
    },
    "maximum_resistance": {
        "name": "Challenge",
        "description": "The user is skeptical, pushing back, or presenting a counterargument. They are not yet convinced.",
        "guide_orientation": "Engage the challenge directly. Acknowledge what is valid in the pushback. Reframe rather than dismiss.",
    },
    "yielding": {
        "name": "Acceptance",
        "description": "The user is beginning to accept or engage with the concept. They are moving from resistance toward understanding.",
        "guide_orientation": "Reinforce the shift without over-explaining. Help the user consolidate what they are coming to understand.",
    },
    "integration": {
        "name": "Synthesis",
        "description": "The user understands the concept and is connecting it to their broader knowledge. They are asking how it relates to other things they know.",
        "guide_orientation": "Help the user build connections. Encourage them to apply or extend the concept. The voice is more exploratory and collaborative.",
    },
}

INTENSITY_DEFINITIONS = {
    "acute": {
        "name": "Terse",
        "description": "Short, direct exchange. The user asks a brief question with minimal context. The assistant responds concisely.",
    },
    "moderate": {
        "name": "Detailed",
        "description": "Fuller exchange. The user provides context and asks a substantive question. The assistant gives a thorough, well-structured response.",
    },
}
