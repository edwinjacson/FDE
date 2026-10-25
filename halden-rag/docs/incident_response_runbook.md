# Incident Response Runbook
Last updated: 11 November 2024
Owner: Platform Operations

## Severity levels
| Severity | Definition                                   | Response target         |
|----------|----------------------------------------------|-------------------------|
| SEV1     | Total outage or data breach affecting customers | Within 15 minutes, 24/7 |
| SEV2     | Major feature degraded, workaround exists    | Within 1 hour           |
| SEV3     | Minor issue, no customer impact              | Next business day       |

## Communications
For SEV1 incidents the public status page must be updated every 30 minutes until resolution. For SEV2 incidents, update the status page at least every 2 hours. The incident commander owns all external communication.

## On-call
The on-call rota is weekly, with handover on Mondays at 10:00. The primary on-call engineer acknowledges alerts; the secondary is the escalation point.

## After the incident
A post-incident review must be completed within 5 working days of resolution for every SEV1 and SEV2. Reviews are blameless and published to the engineering wiki.
