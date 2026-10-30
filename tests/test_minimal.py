import csv,json,tempfile,unittest
from pathlib import Path
import numpy as np
from massive_eval.core import scores,selective,curve,threshold_at_coverage
from massive_eval.raw_audit import parse_annotation
from massive_eval.round11 import events,holm,bootstrap_counts
from massive_eval.io import load_to_jsonl
from massive_eval.cli import demo
from massive_eval.evaluation import paired,evaluate,read_predictions

def read_csv(path):
    with path.open(newline="") as f:return list(csv.DictReader(f))

class MinimalTests(unittest.TestCase):
    def test_annotation_retains_values(self):
        slots,text,errors=parse_annotation('set alarm [time : 7 pm]')
        self.assertEqual(text,'set alarm 7 pm');self.assertEqual(slots[0]['type'],'time');self.assertFalse(errors)
    def test_bad_brackets(self):
        self.assertTrue(parse_annotation('bad [time 7]')[2])
    def test_empty_acceptance_is_undefined(self):
        self.assertIsNone(selective([True],[.2],.9)['risk'])
    def test_threshold_ties_are_kept(self):
        t=threshold_at_coverage([.9,.7,.7,.1],.5)
        self.assertEqual(selective(np.array([True,True,False,False]),[.9,.7,.7,.1],t)['accepted'],3)
    def test_curve_keeps_complete_ties(self):
        c,r,a=curve([True,False,True],[.5,.5,.1])
        np.testing.assert_allclose(c,[2/3,1]);np.testing.assert_allclose(r,[.5,1/3]);self.assertAlmostEqual(a,4/9)
    def test_entropy_direction(self):
        s=scores(np.array([[.9,.1],[.5,.5]]));self.assertGreater(s['negative_entropy'][0],s['negative_entropy'][1])
    def test_cluster_resampling_keeps_related_rows(self):
        x=bootstrap_counts(np.array([[1,1],[1,1],[1,0]],float),['a','a','b'],50,1)
        self.assertTrue(set(x[:,1])<={0,2,4})
    def test_union_event(self):
        e=events(np.array([[1,1,1],[1,0,1]],bool),np.array([[0,0,0],[0,0,0]],bool))
        np.testing.assert_equal(e['mixed_accepted_error'],[False,True])
    def test_holm(self):np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])
    def test_stream_loader_filters_without_rewriting(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);r={'id':'1','locale':'ja-JP','partition':'dev','utt':'架空のテスト','intent':'toy'};(p/'x.jsonl').write_text(json.dumps(r)+'\n')
            x=load_to_jsonl(p/'x.jsonl',p/'out.jsonl',['ja-JP'],'dev');self.assertEqual(x['records'],1);self.assertEqual(json.loads((p/'out.jsonl').read_text()),r)
    def test_demo_end_to_end_and_id_pairing_not_row_order(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);demo(p);before=read_csv(p/'paired.csv');rows=read_predictions(p/'predictions.csv')
            with (p/'shuffled.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows[::-1])
            paired(p/'shuffled.csv',p/'thresholds.json',p/'again.csv');self.assertEqual(before,read_csv(p/'again.csv'))
    def test_refuses_test_selected_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);demo(p);ts=json.loads((p/'thresholds.json').read_text());ts[0]['selected_on']='test';(p/'bad.json').write_text(json.dumps(ts))
            with self.assertRaises(ValueError):evaluate(p/'predictions.csv',p/'bad.json',p/'bad.csv')
    def test_duplicate_prediction_id_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);demo(p);s=(p/'predictions.csv').read_text().splitlines();(p/'dup.csv').write_text('\n'.join(s+[s[1]])+'\n')
            with self.assertRaises(ValueError):read_predictions(p/'dup.csv')
if __name__=='__main__':unittest.main()
