"""Environment-only credential loading; fixtures never access real secrets."""
import os
import unittest
from unittest.mock import patch
import server


class PublicConfigTests(unittest.TestCase):
    def test_key_is_taken_from_explicit_environment(self):
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': ' offline-test-only '}, clear=True):
            self.assertEqual(server.load_api_key(), 'offline-test-only')

    def test_missing_key_does_not_read_a_personal_credential_file(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(server.Path, 'read_text', side_effect=AssertionError('Must not read credentials')):
            with self.assertRaisesRegex(RuntimeError, 'DEEPSEEK_API_KEY'):
                server.load_api_key()


if __name__ == '__main__': unittest.main()
