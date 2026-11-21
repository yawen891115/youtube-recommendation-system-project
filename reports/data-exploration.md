# Data exploration

The dataset holds 100,000 real plays by 4,088 users over a catalog of 7,583 short videos (5,821 of them played; 44 primary-tag categories), from 2022-04-09T03:59:02.423000+00:00 to 2022-05-08T14:30:29.614000+00:00. Source: KuaiRand-Pure (Kuaishou), CC BY-SA 4.0; the subset rules are in `data/README.md`.

Like rate: 3.09%. Share rate: 0.18%. Long-view rate: 71.97%. Mean watch duration: 47.4 seconds. Plays longer than the video (rewatch loops): 33,345. Plays dated before upload: 0.

## Structure a pre-watch ranker can use (training split)

| Check | Value |
| --- | ---: |
| Share of plays on the top 1% of videos | 19.2% |
| Video popularity rank correlation, first vs second half of training | 0.220 |
| Plays of a creator the user had already played | 1.7% |
| Mean share of a user's plays in their top category (users with 5+ plays) | 38.3% |
| Test plays by users seen in training | 91.6% |
| Test plays of videos seen in training | 96.8% |

## Split distributions

- train: 64,000 plays, 3.08% liked, 2022-04-09T03:59:02.423000+00:00 to 2022-04-15T09:12:58.080000+00:00.
- validation: 16,000 plays, 3.54% liked, 2022-04-15T09:13:20.752000+00:00 to 2022-04-21T14:37:29.387000+00:00.
- test: 20,000 plays, 2.78% liked, 2022-04-21T14:39:34.634000+00:00 to 2022-05-08T14:30:29.614000+00:00.
