"""
Telemetry and Observability for Brand Guardian (Local Mode).

Configures local application logging without requiring Azure Application Insights.
"""

import logging

logger = logging.getLogger("brand-guardian-telemetry")

def setup_telemetry():
    """
    Configures local logging. If OpenTelemetry / Azure Monitor is desired later,
    it can be hooked in here.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger.info("Local Telemetry & Structured Logging initialized.")
