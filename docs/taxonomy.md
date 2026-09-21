# Active taxonomy

For annotation, follow the ordered decision procedure in
`configs/annotation_prompt.md`: international scope first, domestic boundaries,
group/institution fallback, the explicit other decision, primary tie-break, then
optional secondary. Use `docs/annotation_handoff.md` for a fresh versioned run.

The taxonomy is version 3 (23 classes): `blm` is now `racism`, `anti-fascism and extremism`
is a separate class, and the broad anti-government class has been removed.
Existing annotation files must be reviewed before use with this version.

Version 3 adds `crime, violence and victim justice` for crime, victim solidarity,
and demands for justice, and `civil liberties and censorship` for civic freedoms
and restrictions on them. `Unjust law enforcement` retains its existing name
and definition. Demanding justice for a crime does not itself allege police or
judicial misconduct; the boundary rules distinguish those cases explicitly.

The goal is to identify the substantive domain of each protest with one primary
label and, optionally, one secondary label in `alternative_labels`. Include a
secondary label when the note supports another domain as a secondary theme or
a defensible alternative interpretation, applying the same rules to both labels.
Otherwise leave it empty. Use the stated
demand to identify it; when the demand is vague or unspecified, use the protesting
group's role or the affected institution as the default. An explicit grievance
pointing to another domain overrides that default: students protesting for
funding belong to education, while students protesting corruption belong to
democratic institutions, corruption, elections, and reforms. Apply the
category-specific boundary rules where domains overlap.

The canonical labels, definitions, and boundary rules are implemented in
`src/protest_classifier/taxonomy.py`. The annotator-facing version is
`configs/annotation_prompt.md`. Keep these two files synchronized whenever the
taxonomy is revised; changing the taxonomy requires new dev and locked-test
annotations.
