#!/usr/bin/env python
"""
Finetuning script for GPT financial agent.
This script handles the finetuning process using OpenAI's API.
"""

import os
import json
import argparse
import logging
import time
import openai
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_to_openai_format(finetuning_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert finetuning data to OpenAI's expected format.
    
    Args:
        finetuning_data: List of training examples
        
    Returns:
        List of examples in OpenAI's format
    """
    openai_format = []
    
    for example in finetuning_data:
        messages = example.get("messages", [])
        formatted_messages = []
        
        for message in messages:
            role = message.get("role", "").lower()
            # Convert 'human' to 'user' for OpenAI
            if role == "human":
                role = "user"
            # Convert 'assistant' to 'assistant' (no change needed)
            
            formatted_messages.append({
                "role": role,
                "content": message.get("content", "")
            })
        
        if formatted_messages:
            openai_format.append({"messages": formatted_messages})
    
    return openai_format

def enhance_with_tool_knowledge(
    finetuning_data: List[Dict[str, Any]], 
    tool_tuning_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Enhance training data with information about available financial tools.
    
    Args:
        finetuning_data: List of training examples
        tool_tuning_data: Tool descriptions and usage information
        
    Returns:
        Enhanced training data with tool information
    """
    enhanced_data = []
    
    # Create a system message about available tools
    tools_info = "You have access to these financial tools:\n"
    for tool_name, tool_info in tool_tuning_data.items():
        description = tool_info.get("description", f"Calculate {tool_name.replace('_', ' ')}")
        tools_info += f"- {tool_name}: {description}\n"
    
    for example in finetuning_data:
        messages = example.get("messages", [])
        
        # Check if a system message already exists
        has_system = any(msg.get("role", "").lower() == "system" for msg in messages)
        
        new_messages = []
        if not has_system:
            # Add system message if none exists
            new_messages.append({
                "role": "system",
                "content": tools_info
            })
        
        # Add original messages
        new_messages.extend(messages)
        
        enhanced_data.append({"messages": new_messages})
    
    return enhanced_data

def finetune_gpt_model(
    finetuning_file_path: str,
    tool_tuning_file_path: str,
    base_model: str = "gpt-4.1-2025-04-14",
    model_name: str = "iso-financial-advisor-v1",
    api_key: Optional[str] = None
) -> str:
    """
    Finetune a GPT model using the provided training data.
    
    Args:
        finetuning_file_path: Path to the finetuning.json file
        tool_tuning_file_path: Path to the tool_tuning.json file
        base_model: Base model to finetune from
        model_name: Name to give the finetuned model
        api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
        
    Returns:
        ID of the created finetuning job
    """
    # Set API key
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
    
    openai.api_key = api_key
    
    # Load finetuning data
    logger.info(f"Loading finetuning data from {finetuning_file_path}")
    try:
        with open(finetuning_file_path, 'r') as f:
            finetuning_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading finetuning data: {str(e)}")
        raise ValueError(f"Error loading finetuning data: {str(e)}")
    
    # Load tool tuning data
    logger.info(f"Loading tool tuning data from {tool_tuning_file_path}")
    try:
        with open(tool_tuning_file_path, 'r') as f:
            tool_tuning_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading tool tuning data: {str(e)}")
        raise ValueError(f"Error loading tool tuning data: {str(e)}")
    
    # Convert data to OpenAI format
    openai_format = convert_to_openai_format(finetuning_data)
    
    # Enhance data with tool information
    enhanced_data = enhance_with_tool_knowledge(openai_format, tool_tuning_data)
    
    # Save formatted training data to a temporary file
    temp_file_path = "temp_training_data.jsonl"
    with open(temp_file_path, 'w') as f:
        for example in enhanced_data:
            f.write(json.dumps(example) + '\n')
    
    logger.info(f"Saved {len(enhanced_data)} training examples to {temp_file_path}")
    
    # Upload training file
    logger.info("Uploading training file to OpenAI")
    try:
        with open(temp_file_path, 'rb') as f:
            response = openai.files.create(
                file=f,
                purpose="fine-tune"
            )
        file_id = response.id
        logger.info(f"File uploaded with ID: {file_id}")
    except Exception as e:
        logger.error(f"Error uploading training file: {str(e)}")
        raise ValueError(f"Error uploading training file: {str(e)}")
    
    # Wait for file processing to complete
    logger.info("Waiting for file processing to complete...")
    time.sleep(5)  # Give some time for the file to be processed
    
    # Create finetuning job
    logger.info(f"Creating finetuning job for model {model_name} based on {base_model}")
    try:
        response = openai.fine_tuning.jobs.create(
            training_file=file_id,
            model=base_model,
            suffix=model_name,
            hyperparameters={
                "n_epochs": 3  # Adjust as needed
            }
        )
        job_id = response.id
        logger.info(f"Finetuning job created successfully with ID: {job_id}")
        
        # Clean up temporary file
        os.remove(temp_file_path)
        
        return job_id
    except Exception as e:
        logger.error(f"Error creating finetuning job: {str(e)}")
        raise ValueError(f"Error creating finetuning job: {str(e)}")

def check_finetuning_status(job_id: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Check the status of a finetuning job.
    
    Args:
        job_id: ID of the finetuning job
        api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
        
    Returns:
        Status information about the finetuning job
    """
    # Set API key
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
    
    openai.api_key = api_key
    
    logger.info(f"Checking status of finetuning job {job_id}")
    try:
        response = openai.fine_tuning.jobs.retrieve(job_id)
        
        status_info = {
            "id": response.id,
            "status": response.status,
            "model": response.model,
            "created_at": response.created_at,
            "finished_at": response.finished_at,
            "fine_tuned_model": response.fine_tuned_model
        }
        logger.info(f"Finetuning job status: {status_info['status']}")
        return status_info
    except Exception as e:
        logger.error(f"Error checking finetuning status: {str(e)}")
        return {"error": str(e)}

def save_model_info(model_info: Dict[str, Any], output_file: str) -> None:
    """
    Save model information to a file.
    
    Args:
        model_info: Model information dictionary
        output_file: Path to save the information
    """
    try:
        with open(output_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        logger.info(f"Model information saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving model information: {str(e)}")

def main():
    """Main function to run the finetuning script."""
    parser = argparse.ArgumentParser(description="Finetune GPT for financial analysis")
    parser.add_argument("--finetuning-file", type=str, default="../finetuning/finetuning.json",
                        help="Path to finetuning data file")
    parser.add_argument("--tool-tuning-file", type=str, default="../finetuning/tool_tuning.json",
                        help="Path to tool tuning data file")
    parser.add_argument("--base-model", type=str, default="gpt-4.1-2025-04-14",
                        help="Base model to finetune from")
    parser.add_argument("--model-name", type=str, default="iso-financial-advisor-v1",
                        help="Name for the finetuned model")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (optional)")
    parser.add_argument("--output-file", type=str, default="model_info.json",
                        help="Output file to save model information")
    parser.add_argument("--check-job", type=str, help="Check status of a job ID")
    
    args = parser.parse_args()
    
    # If checking job status
    if args.check_job:
        status = check_finetuning_status(args.check_job, args.api_key)
        print(json.dumps(status, indent=2))
        if "id" in status and "status" in status:
            save_model_info(status, args.output_file)
        return
    
    # Otherwise, start a new finetuning job
    try:
        job_id = finetune_gpt_model(
            args.finetuning_file,
            args.tool_tuning_file,
            args.base_model,
            args.model_name,
            args.api_key
        )
        
        print(f"Finetuning job started with ID: {job_id}")
        
        # Get initial status and save
        status = check_finetuning_status(job_id, args.api_key)
        save_model_info(status, args.output_file)
        
    except Exception as e:
        logger.error(f"Finetuning error: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()