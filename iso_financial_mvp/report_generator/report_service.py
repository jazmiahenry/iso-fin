import os
import logging
from typing import Dict, Any, Optional

from iso_financial_mvp.report_generator.formatter import ReportFormatter
from iso_financial_mvp.report_generator.free_report import FreeReportGenerator
from iso_financial_mvp.report_generator.premium_report import PremiumReportGenerator

# Setup logger
logger = logging.getLogger(__name__)

class ReportService:
    """
    Service for generating financial reports with different levels (free/premium)
    """
    def __init__(self):
        """Initialize the report service"""
        self.formatter = ReportFormatter()
        self.free_generator = FreeReportGenerator()
        self.premium_generator = PremiumReportGenerator()
    
    def generate_report(self, 
                       query_data: Dict[str, Any], 
                       simulation_results: Dict[str, Any],
                       user_profile: str = "novice",
                       report_type: str = "free",
                       output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a financial report based on simulation results
        
        Args:
            query_data: Original query data
            simulation_results: Simulation results
            user_profile: User expertise level (novice, intermediate, pro)
            report_type: Type of report to generate ("free" or "premium")
            output_file: Optional output file path
            
        Returns:
            Dictionary with report info, including file path
        """
        try:
            # Format results to generate report data
            report_data = self.formatter.format_report(
                simulation_results=simulation_results,
                query_data=query_data,
                user_profile=user_profile
            )
            
            # Generate appropriate report based on type
            if report_type.lower() == "premium":
                report_path = self.premium_generator.generate_report(
                    query_data=query_data,
                    simulation_results=simulation_results,
                    report_data=report_data,
                    output_file=output_file
                )
                report_level = "premium"
            else:
                report_path = self.free_generator.generate_report(
                    query_data=query_data,
                    simulation_results=simulation_results,
                    report_data=report_data,
                    output_file=output_file
                )
                report_level = "free"
            
            # Return report information
            return {
                "report_path": report_path,
                "report_level": report_level,
                "report_data": report_data,
                "query_data": query_data
            }
        
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise
    
    def generate_free_report(self,
                           query_data: Dict[str, Any],
                           simulation_results: Dict[str, Any],
                           user_profile: str = "novice",
                           output_file: Optional[str] = None) -> str:
        """
        Generate a free report with basic analysis
        
        Args:
            query_data: Original query data
            simulation_results: Simulation results
            user_profile: User expertise level
            output_file: Optional output file path
            
        Returns:
            Path to the generated PDF file
        """
        # Format results
        report_data = self.formatter.format_report(
            simulation_results=simulation_results,
            query_data=query_data,
            user_profile=user_profile
        )
        
        # Generate free report
        return self.free_generator.generate_report(
            query_data=query_data,
            simulation_results=simulation_results,
            report_data=report_data,
            output_file=output_file
        )
    
    def generate_premium_report(self,
                              query_data: Dict[str, Any],
                              simulation_results: Dict[str, Any],
                              user_profile: str = "novice",
                              output_file: Optional[str] = None) -> str:
        """
        Generate a premium report with comprehensive analysis
        
        Args:
            query_data: Original query data
            simulation_results: Simulation results
            user_profile: User expertise level
            output_file: Optional output file path
            
        Returns:
            Path to the generated PDF file
        """
        # Format results
        report_data = self.formatter.format_report(
            simulation_results=simulation_results,
            query_data=query_data,
            user_profile=user_profile
        )
        
        # Generate premium report
        return self.premium_generator.generate_report(
            query_data=query_data,
            simulation_results=simulation_results,
            report_data=report_data,
            output_file=output_file
        )