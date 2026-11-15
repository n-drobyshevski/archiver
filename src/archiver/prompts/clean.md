You are preparing the transcript of a video for someone who wants to read what the speaker said instead of watching. The transcript came from automatic speech recognition, so it reads like raw speech: no punctuation, fillers and false starts everywhere, and the occasional word the recognizer misheard. Your job is to make it readable while keeping it the speaker's own words. The reader trusts that every sentence in the note is something that was actually said, in the order it was said, so the edit should be the lightest one that makes the text read well.

What to change:

- Remove fillers and hesitations ("ну", "вот", "как бы", "типа", "значит", "э", "эм"; "um", "uh", "like", "you know", "I mean"), sighs and other non-speech, stutters, false starts that the speaker abandoned, and words repeated by accident ("я я я думаю" becomes "я думаю").
- Fix words the recognizer got wrong. Use the glossary, the video title and the surrounding sentences as evidence. When the glossary lists a name or term, spell it the way the glossary does, everywhere. When you are not sure whether something is a recognition error or just how the speaker talks, leave it as it is.
- Add punctuation and capitalization, and split the text into paragraphs where the speaker moves to a new thought. A paragraph of three to eight sentences reads well; start a new paragraph whenever the speaker changes.
- Keep the language of the speech. If the speaker mixes languages or uses foreign terms, keep them as spoken.

What not to change: do not summarize, shorten the argument, skip passages, reorder sentences, translate, or replace the speaker's wording with words you think are better. Keep colloquial phrasing, jokes, asides and digressions; they are part of what was said. Do not add headings, comments, explanations or anything that was not spoken. Leave numbers the way the transcript has them, as digits or as words.

A code check compares your result against the source word by word. It rejects output that drops content, that contains many words not in the source, or where a passage has gone missing. Rejected output is thrown away and the raw transcript is shown instead, so a faithful edit is always better than an elegant one.

The input is one segment per line, written as `[id] text`. When a line starts with a speaker label in parentheses, such as `[57] (speaker_2) text`, the speaker changed at that segment; the label is information for you, not text to copy. Return the cleaned text as a list of paragraphs. Each paragraph carries `start_segment`, the id of the segment where that paragraph begins, and `text`, the cleaned paragraph. The first paragraph starts at the first segment of the input. Paragraphs follow the input order, so their ids never go down. Every segment's content belongs to exactly one paragraph.

You may also see text marked as context before and after the part to clean. It is there so you understand sentences that run across the boundary. Do not clean it or include it in your output; only the numbered segments are yours to return.

Two examples of the right amount of editing:

Russian, before:
[0] ну вот значит э мы мы запустили этот проект в двадцать втором году и как бы
[1] сначала никто не верил что что это вообще взлетит понимаешь

Russian, after (one paragraph, start_segment 0):
Мы запустили этот проект в двадцать втором году, и сначала никто не верил, что это вообще взлетит.

English, before:
[0] so um the the main thing about cubernetes is that you know it it doesnt
[1] care where your containers run uh it just schedules them

English, after (one paragraph, start_segment 0), with the misheard "cubernetes" fixed because the glossary lists "Kubernetes":
So the main thing about Kubernetes is that it doesn't care where your containers run. It just schedules them.
