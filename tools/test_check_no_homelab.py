"""tools/check_no_homelab.py: the guard that keeps private addresses and home-lab names out of this public repository."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_no_homelab as g  # noqa: E402


def diff(path, *lines):
    return "diff --git a/%s b/%s\n--- a/%s\n+++ b/%s\n@@ -0,0 +1 @@\n%s\n" % (path, path, path, path, "\n".join("+" + x for x in lines))


class Guard(unittest.TestCase):
    def test_flags_private_addresses(self):
        for ip in ("192.168.1.130", "10.0.0.5", "172.16.4.9", "172.31.255.1", "169.254.1.1"):
            self.assertTrue(g.scan(diff("docs/x.md", "serve on %s:7840" % ip)), ip)

    def test_documentation_ranges_pass(self):
        for ip in ("192.0.2.10", "198.51.100.7", "203.0.113.9", "127.0.0.1", "0.0.0.0", "172.32.0.1"):
            self.assertEqual(g.scan(diff("docs/x.md", "serve on %s:7840" % ip)), [], ip)

    def test_flags_names_paths_and_gpu_uuids(self):
        for text in ("the T340 host", "ssh dadbodnerd@host", "C:\\AIO-strata\\build", "/mnt/models/strata", "the stream PC",
                     "GPU-d526893c-b378-4be2-4e99-71848237f300", "D:/AIStuff/x"):
            self.assertTrue(g.scan(diff("docs/x.md", text)), text)

    def test_placeholders_pass(self):
        for text in ("<model-dir>\\pack", "GPU-00000000-0000-0000-0000-000000000000", "host-b at 192.0.2.11", "the Windows host"):
            self.assertEqual(g.scan(diff("docs/x.md", text)), [], text)

    def test_only_added_lines_and_the_opt_out(self):
        removed = "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +0,0 @@\n-old 192.168.1.5\n"
        self.assertEqual(g.scan(removed), [])
        self.assertEqual(g.scan(diff("docs/x.md", "fixture 192.168.1.5 homelab-ok")), [])


if __name__ == "__main__":
    unittest.main()
