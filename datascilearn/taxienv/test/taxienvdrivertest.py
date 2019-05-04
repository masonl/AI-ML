'''
Created on May 3, 2019

@author: Mason
'''

from  taxienv.taxienvdriver import TaxiEnvDriver 

import unittest


class Test(unittest.TestCase):
    class Actual():
        ACTION_COUNT = 6
        STATE_COUNT = 500
        MODE = 'ansi'
        RENDER = '+---------+\n|\x1b[35mR\x1b[0m: | : :G|\[96 chars]\n\n'

    def setUp(self):
        self._envd = TaxiEnvDriver()


    def tearDown(self):
        self._envd.close()


    def testActionCount(self):
        self.assertEqual(self._envd.action_count,
                         Test.Actual.ACTION_COUNT, 'Action space count is not correct.')

    def testStateCount(self):
        self.assertEqual(self._envd.state_count, Test.Actual.STATE_COUNT, 'State space count is not correct.')

    def testRender(self):
        self._envd.set_state()
        self.assertEqual(self._envd.render(mode=Test.Actual.MODE), Test.Actual.RENDER, 'Taxi Env driver render is not correct.')
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()