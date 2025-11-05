"""
Jinja2-based prompt template system for LLM analysis.

Provides version-controlled, file-based prompt templates with:
- Template inheritance and composition
- Secure template rendering with sandboxing
- Convenient helper methods for common analysis tasks
"""

from pathlib import Path
from typing import Any

from jinja2 import FileSystemLoader, TemplateNotFound, select_autoescape
from jinja2.sandbox import SandboxedEnvironment


class PromptManager:
    """Manages Jinja2 prompt templates for medical analysis."""

    def __init__(self, templates_dir: Path = None):
        """
        Initialize prompt manager.

        Args:
            templates_dir: Path to templates directory.
                          Defaults to prompts/ in this package.
        """
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "prompts"

        self.templates_dir = Path(templates_dir)

        # Sandboxed environment prevents code execution in templates
        # (defense-in-depth, in case templates are ever loaded from external sources)
        self.env = SandboxedEnvironment(
            loader=FileSystemLoader(self.templates_dir),
            autoescape=select_autoescape(
                enabled_extensions=("html", "xml", "jinja2"),
                default_for_string=True,
            ),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render_prompt(self, template_name: str, **context: Any) -> str:
        """
        Render a prompt template with context variables.

        Args:
            template_name: Template filename (e.g., "flow_limitation/analysis.jinja2")
            **context: Template variables to render

        Returns:
            Rendered prompt string

        Raises:
            TemplateNotFound: If the template file doesn't exist
            ValueError: If template rendering fails

        Example:
            >>> pm = PromptManager()
            >>> prompt = pm.render_prompt(
            ...     "flow_limitation/analysis.jinja2",
            ...     breath_descriptions=[...],
            ...     reference_patterns={...}
            ... )
        """
        try:
            template = self.env.get_template(template_name)
            return template.render(**context)
        except TemplateNotFound as e:
            raise TemplateNotFound(
                f"Template '{template_name}' not found in {self.templates_dir}"
            ) from e

    def render_flow_limitation_analysis(
        self, breath_descriptions: list, reference_patterns: dict
    ) -> str:
        """
        Convenience method for flow limitation analysis.

        Args:
            breath_descriptions: List of breath description dicts with visual
                               descriptions, metrics, and timestamps
            reference_patterns: Dict mapping class numbers to pattern definitions

        Returns:
            Rendered analysis prompt
        """
        return self.render_prompt(
            "flow_limitation/analysis.jinja2",
            breath_descriptions=breath_descriptions,
            reference_patterns=reference_patterns,
        )

    def list_templates(self) -> list[str]:
        """
        List all available templates.

        Returns:
            List of template paths relative to templates_dir
        """
        templates = []
        for template_file in self.templates_dir.rglob("*.jinja2"):
            relative_path = template_file.relative_to(self.templates_dir)
            templates.append(str(relative_path))
        return sorted(templates)
