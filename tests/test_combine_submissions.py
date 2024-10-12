import sys
import pytest
sys.path.append('scripts')

from combine_submissions import combine_submissions


@pytest.mark.parametrize('sub_1, sub_2, expected_combination', [
    [dict(a=[dict(attempt_1=[1], attempt_2=[])]), dict(a=[dict(attempt_1=[2], attempt_2=[])]), dict(a=[dict(attempt_1=[1], attempt_2=[2])])],
    [dict(a=[dict(attempt_1=[1], attempt_2=[])]), dict(a=[dict(attempt_1=[1], attempt_2=[2])]), dict(a=[dict(attempt_1=[1], attempt_2=[2])])],
    [dict(a=[dict(attempt_1=[1], attempt_2=[3])]), dict(a=[dict(attempt_1=[2], attempt_2=[])]), dict(a=[dict(attempt_1=[1], attempt_2=[2])])],
    [dict(a=[dict(attempt_1=[1], attempt_2=[3])]), dict(a=[dict(attempt_1=[1], attempt_2=[2])]), dict(a=[dict(attempt_1=[1], attempt_2=[2])])],
    [dict(a=[dict(attempt_1=[1], attempt_2=[3])]), dict(a=[dict(attempt_1=[], attempt_2=[])]), dict(a=[dict(attempt_1=[1], attempt_2=[3])])],
    [dict(a=[dict(attempt_1=[], attempt_2=[])]), dict(a=[dict(attempt_1=[1], attempt_2=[2])]), dict(a=[dict(attempt_1=[1], attempt_2=[2])])],
    [dict(b=[dict(attempt_1=[1], attempt_2=[])]), dict(a=[dict(attempt_1=[1], attempt_2=[2])]), dict(a=[dict(attempt_1=[1], attempt_2=[2])],
                                                                                                     b=[dict(attempt_1=[1], attempt_2=[])])],
])
def test_combine_submissions_when_giving_preference_to_second_submission(sub_1, sub_2, expected_combination):
    combined_sub = combine_submissions(sub_1, sub_2, give_preference_to_second_submission_on_second_attempt=True)
    assert combined_sub == expected_combination


@pytest.mark.parametrize('sub_1, sub_2, expected_combination', [
    [dict(a=[dict(attempt_1=[1], attempt_2=[])]), dict(a=[dict(attempt_1=[2], attempt_2=[])]), dict(a=[dict(attempt_1=[1], attempt_2=[2])])],
    [dict(a=[dict(attempt_1=[1], attempt_2=[])]), dict(a=[dict(attempt_1=[1], attempt_2=[2])]), dict(a=[dict(attempt_1=[1], attempt_2=[2])])],
    [dict(a=[dict(attempt_1=[1], attempt_2=[3])]), dict(a=[dict(attempt_1=[2], attempt_2=[])]), dict(a=[dict(attempt_1=[1], attempt_2=[3])])],
    [dict(a=[dict(attempt_1=[1], attempt_2=[3])]), dict(a=[dict(attempt_1=[1], attempt_2=[2])]), dict(a=[dict(attempt_1=[1], attempt_2=[3])])],
    [dict(a=[dict(attempt_1=[1], attempt_2=[3])]), dict(a=[dict(attempt_1=[], attempt_2=[])]), dict(a=[dict(attempt_1=[1], attempt_2=[3])])],
    [dict(a=[dict(attempt_1=[], attempt_2=[])]), dict(a=[dict(attempt_1=[1], attempt_2=[2])]), dict(a=[dict(attempt_1=[1], attempt_2=[2])])],
    [dict(b=[dict(attempt_1=[1], attempt_2=[])]), dict(a=[dict(attempt_1=[1], attempt_2=[2])]), dict(a=[dict(attempt_1=[1], attempt_2=[2])],
                                                                                                     b=[dict(attempt_1=[1], attempt_2=[])])],
])
def test_combine_submissions_when_giving_preference_to_first_submission(sub_1, sub_2, expected_combination):
    combined_sub = combine_submissions(sub_1, sub_2, give_preference_to_second_submission_on_second_attempt=False)
    assert combined_sub == expected_combination
