"""Run with python -m unittest discover -s tests; no model calls are sent."""
import unittest
from unittest.mock import patch

from Method import utils


FIELDS = [('Reasoning Process', 'text'), ('Revised Hypothesis', 'text')]
VALID = ('**Revised Hypothesis starts:** A revised hypothesis. **Revised Hypothesis ends**\n'
         '**Reasoning Process starts:** A rationale. **Reasoning Process ends**')


class StructuredExtractionTest(unittest.TestCase):
    def extract(self, **kwargs):
        return utils.llm_generation_with_extraction('prompt', 'model', None, FIELDS, **kwargs)

    def test_refusal_exhaustion_raises_instead_of_returning_none(self):
        with patch.object(utils, 'llm_generation', return_value='I cannot provide a hypothesis.') as generate:
            with self.assertRaisesRegex(ValueError, 'missing fields: Reasoning Process, Revised Hypothesis'):
                self.extract(max_retries=3)
            self.assertEqual(generate.call_count, 3)

    def test_partial_fields_are_not_returned(self):
        text = '**Reasoning Process starts:** A rationale. **Reasoning Process ends**'
        with patch.object(utils, 'llm_generation', return_value=text):
            with self.assertRaisesRegex(ValueError, 'missing fields: Revised Hypothesis'):
                self.extract(max_retries=2)

    def test_later_valid_response_succeeds(self):
        with patch.object(utils, 'llm_generation', side_effect=['Unstructured response', VALID]) as generate:
            self.assertEqual(self.extract(max_retries=3), {
                'Reasoning Process': 'A rationale.', 'Revised Hypothesis': 'A revised hypothesis.'})
            self.assertEqual(generate.call_count, 2)

    def test_legacy_wrapper_propagates_failure(self):
        with patch.object(utils, 'llm_generation', return_value='No hypothesis.') as generate:
            with self.assertRaisesRegex(ValueError, '15 attempts'):
                utils.llm_generation_while_loop('prompt', 'model', None,
                    if_structured_generation=True, template=['Reasoning Process:', 'Revised Hypothesis:'])
            self.assertEqual(generate.call_count, 15)

    def test_extraction_exception_preserves_cause(self):
        error = RuntimeError('parser failed')
        with patch.object(utils, 'llm_generation', return_value=VALID), \
                patch.object(utils, 'extract_field', side_effect=error):
            with self.assertRaisesRegex(ValueError, '2 attempts') as caught:
                self.extract(max_retries=2)
            self.assertIs(caught.exception.__cause__, error)

    def test_false_and_zero_are_valid_fields(self):
        text = 'Answer starts: no Answer ends\nScore starts: 0 Score ends'
        with patch.object(utils, 'llm_generation', return_value=text):
            self.assertEqual(utils.llm_generation_with_extraction('prompt', 'model', None,
                [('Answer', 'bool'), ('Score', 'number')]), {'Answer': False, 'Score': 0})

    def test_invalid_retry_budget_is_rejected(self):
        with patch.object(utils, 'llm_generation') as generate:
            with self.assertRaisesRegex(ValueError, 'max_retries'):
                self.extract(max_retries=0)
            generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
