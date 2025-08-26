"""Educational AI module for AI Bull Ford.

This module provides comprehensive educational AI capabilities including:
- Personalized learning systems
- Intelligent tutoring systems
- Content generation and curation
- Assessment and evaluation
- Learning analytics
- Adaptive learning paths
- Educational content recommendation
"""

import asyncio
import logging
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class LearningStyle(Enum):
    """Different learning styles."""
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READING_WRITING = "reading_writing"
    MULTIMODAL = "multimodal"


class DifficultyLevel(Enum):
    """Content difficulty levels."""
    BEGINNER = "beginner"
    ELEMENTARY = "elementary"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ContentType(Enum):
    """Types of educational content."""
    TEXT = "text"
    VIDEO = "video"
    AUDIO = "audio"
    INTERACTIVE = "interactive"
    SIMULATION = "simulation"
    QUIZ = "quiz"
    EXERCISE = "exercise"
    PROJECT = "project"
    GAME = "game"
    PRESENTATION = "presentation"


class AssessmentType(Enum):
    """Types of assessments."""
    FORMATIVE = "formative"
    SUMMATIVE = "summative"
    DIAGNOSTIC = "diagnostic"
    PEER = "peer"
    SELF = "self"
    ADAPTIVE = "adaptive"
    PORTFOLIO = "portfolio"


class LearningObjective(Enum):
    """Learning objectives taxonomy."""
    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


class EngagementLevel(Enum):
    """Student engagement levels."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class EducationalConfig:
    """Configuration for educational AI systems."""
    institution_id: str = "school_001"
    adaptive_learning_enabled: bool = True
    personalization_enabled: bool = True
    analytics_enabled: bool = True
    content_generation_enabled: bool = True
    real_time_feedback: bool = True
    privacy_compliant: bool = True
    data_encryption: bool = True
    min_confidence_threshold: float = 0.7
    max_difficulty_jump: int = 2  # Maximum difficulty level jump
    session_timeout_minutes: int = 60
    progress_save_interval: int = 300  # seconds
    logging_enabled: bool = True


@dataclass
class LearnerProfile:
    """Comprehensive learner profile."""
    learner_id: str
    name: str
    age: int
    grade_level: str
    learning_style: LearningStyle
    preferred_content_types: List[ContentType] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    current_level: Dict[str, DifficultyLevel] = field(default_factory=dict)
    learning_pace: str = "normal"  # slow, normal, fast
    attention_span_minutes: int = 30
    motivation_level: float = 0.7  # 0-1 scale
    last_active: datetime = field(default_factory=datetime.now)
    total_study_time: int = 0  # minutes
    achievements: List[str] = field(default_factory=list)


@dataclass
class LearningContent:
    """Educational content item."""
    content_id: str
    title: str
    subject: str
    topic: str
    content_type: ContentType
    difficulty_level: DifficultyLevel
    learning_objectives: List[LearningObjective] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    estimated_duration_minutes: int = 15
    content_data: Any = None  # Actual content (text, video path, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    average_rating: float = 0.0
    suitable_learning_styles: List[LearningStyle] = field(default_factory=list)


@dataclass
class LearningSession:
    """Individual learning session data."""
    session_id: str
    learner_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    content_items: List[str] = field(default_factory=list)  # content_ids
    activities_completed: int = 0
    correct_answers: int = 0
    total_questions: int = 0
    engagement_score: float = 0.0
    time_on_task_minutes: int = 0
    breaks_taken: int = 0
    help_requests: int = 0
    session_notes: str = ""
    completed: bool = False


@dataclass
class Assessment:
    """Assessment or quiz data."""
    assessment_id: str
    title: str
    subject: str
    assessment_type: AssessmentType
    difficulty_level: DifficultyLevel
    questions: List[Dict[str, Any]] = field(default_factory=list)
    total_points: int = 100
    time_limit_minutes: Optional[int] = None
    learning_objectives: List[LearningObjective] = field(default_factory=list)
    adaptive: bool = False
    created_date: datetime = field(default_factory=datetime.now)
    instructions: str = ""
    rubric: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssessmentResult:
    """Result of an assessment."""
    result_id: str
    assessment_id: str
    learner_id: str
    start_time: datetime
    end_time: datetime
    score: float
    percentage: float
    answers: List[Dict[str, Any]] = field(default_factory=list)
    time_taken_minutes: int = 0
    attempts: int = 1
    feedback: str = ""
    strengths_identified: List[str] = field(default_factory=list)
    areas_for_improvement: List[str] = field(default_factory=list)
    next_recommendations: List[str] = field(default_factory=list)


@dataclass
class LearningPath:
    """Personalized learning path."""
    path_id: str
    learner_id: str
    subject: str
    goal: str
    content_sequence: List[str] = field(default_factory=list)  # content_ids
    current_position: int = 0
    estimated_completion_hours: int = 10
    actual_time_spent: int = 0
    progress_percentage: float = 0.0
    adaptive: bool = True
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    prerequisites_met: bool = True


class ContentRecommendationEngine:
    """Recommends educational content based on learner profiles."""
    
    def __init__(self, config: EducationalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.content_library = {}
        self.recommendation_history = {}
        self.learner_interactions = {}
    
    def add_content(self, content: LearningContent) -> None:
        """Add content to the library."""
        try:
            self.content_library[content.content_id] = content
            self.logger.info(f"Added content: {content.title} ({content.content_id})")
        except Exception as e:
            self.logger.error(f"Failed to add content: {e}")
            raise
    
    def recommend_content(self, learner: LearnerProfile, subject: str, 
                         max_recommendations: int = 5) -> List[LearningContent]:
        """Recommend content for a learner."""
        try:
            # Filter content by subject and learner's current level
            suitable_content = []
            learner_level = learner.current_level.get(subject, DifficultyLevel.BEGINNER)
            
            for content in self.content_library.values():
                if content.subject.lower() == subject.lower():
                    # Check difficulty appropriateness
                    level_diff = self._calculate_difficulty_difference(content.difficulty_level, learner_level)
                    if level_diff <= self.config.max_difficulty_jump:
                        # Check learning style compatibility
                        if (not content.suitable_learning_styles or 
                            learner.learning_style in content.suitable_learning_styles):
                            # Check content type preference
                            if (not learner.preferred_content_types or 
                                content.content_type in learner.preferred_content_types):
                                suitable_content.append(content)
            
            # Score and rank content
            scored_content = []
            for content in suitable_content:
                score = self._calculate_content_score(content, learner, subject)
                scored_content.append((content, score))
            
            # Sort by score (descending) and return top recommendations
            scored_content.sort(key=lambda x: x[1], reverse=True)
            recommendations = [content for content, score in scored_content[:max_recommendations]]
            
            # Store recommendation history
            if learner.learner_id not in self.recommendation_history:
                self.recommendation_history[learner.learner_id] = []
            
            self.recommendation_history[learner.learner_id].append({
                'timestamp': datetime.now(),
                'subject': subject,
                'recommendations': [c.content_id for c in recommendations]
            })
            
            self.logger.info(f"Generated {len(recommendations)} recommendations for {learner.learner_id} in {subject}")
            return recommendations
        except Exception as e:
            self.logger.error(f"Failed to recommend content: {e}")
            return []
    
    def _calculate_difficulty_difference(self, content_level: DifficultyLevel, learner_level: DifficultyLevel) -> int:
        """Calculate difficulty difference between content and learner level."""
        levels = [DifficultyLevel.BEGINNER, DifficultyLevel.ELEMENTARY, 
                 DifficultyLevel.INTERMEDIATE, DifficultyLevel.ADVANCED, DifficultyLevel.EXPERT]
        
        content_index = levels.index(content_level)
        learner_index = levels.index(learner_level)
        
        return content_index - learner_index
    
    def _calculate_content_score(self, content: LearningContent, learner: LearnerProfile, subject: str) -> float:
        """Calculate recommendation score for content."""
        score = 0.0
        
        # Base score
        score += 0.5
        
        # Learning style match
        if learner.learning_style in content.suitable_learning_styles:
            score += 0.3
        
        # Content type preference
        if content.content_type in learner.preferred_content_types:
            score += 0.2
        
        # Interest alignment
        content_tags_lower = [tag.lower() for tag in content.tags]
        interest_matches = sum(1 for interest in learner.interests 
                             if interest.lower() in content_tags_lower)
        score += min(0.3, interest_matches * 0.1)
        
        # Difficulty appropriateness
        learner_level = learner.current_level.get(subject, DifficultyLevel.BEGINNER)
        level_diff = self._calculate_difficulty_difference(content.difficulty_level, learner_level)
        
        if level_diff == 0:  # Perfect match
            score += 0.2
        elif level_diff == 1:  # Slightly challenging
            score += 0.15
        elif level_diff == -1:  # Slightly easier (review)
            score += 0.1
        else:  # Too easy or too hard
            score -= 0.1
        
        # Duration appropriateness
        if content.estimated_duration_minutes <= learner.attention_span_minutes:
            score += 0.1
        
        # Content quality (rating)
        score += content.average_rating * 0.1
        
        # Novelty (prefer less used content)
        if content.usage_count < 10:
            score += 0.05
        
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1


class AdaptiveLearningEngine:
    """Manages adaptive learning paths and personalization."""
    
    def __init__(self, config: EducationalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.learning_paths = {}
        self.performance_data = {}
        self.adaptation_rules = {}
    
    def create_learning_path(self, learner: LearnerProfile, subject: str, 
                           goal: str, content_library: Dict[str, LearningContent]) -> LearningPath:
        """Create an adaptive learning path for a learner."""
        try:
            path_id = f"PATH_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{learner.learner_id}_{subject}"
            
            # Filter and sequence content
            relevant_content = [content for content in content_library.values() 
                              if content.subject.lower() == subject.lower()]
            
            # Sort by difficulty and prerequisites
            sequenced_content = self._sequence_content(relevant_content, learner)
            
            # Estimate completion time
            total_duration = sum(content.estimated_duration_minutes for content in sequenced_content)
            estimated_hours = total_duration // 60
            
            # Create milestones
            milestones = self._create_milestones(sequenced_content)
            
            path = LearningPath(
                path_id=path_id,
                learner_id=learner.learner_id,
                subject=subject,
                goal=goal,
                content_sequence=[content.content_id for content in sequenced_content],
                estimated_completion_hours=estimated_hours,
                milestones=milestones
            )
            
            self.learning_paths[path_id] = path
            
            self.logger.info(f"Created learning path {path_id} for {learner.learner_id} in {subject}")
            return path
        except Exception as e:
            self.logger.error(f"Failed to create learning path: {e}")
            raise
    
    def adapt_learning_path(self, path_id: str, performance_data: Dict[str, Any]) -> LearningPath:
        """Adapt learning path based on performance data."""
        try:
            if path_id not in self.learning_paths:
                raise ValueError(f"Learning path {path_id} not found")
            
            path = self.learning_paths[path_id]
            
            # Analyze performance
            avg_score = performance_data.get('average_score', 0.7)
            engagement = performance_data.get('engagement_level', 0.7)
            time_efficiency = performance_data.get('time_efficiency', 1.0)
            
            adaptations_made = []
            
            # Adapt based on performance
            if avg_score < 0.6:  # Struggling
                # Add remedial content or reduce difficulty
                adaptations_made.append("Added remedial content")
                self._add_remedial_content(path)
            elif avg_score > 0.9:  # Excelling
                # Skip some content or increase difficulty
                adaptations_made.append("Increased difficulty level")
                self._increase_difficulty(path)
            
            # Adapt based on engagement
            if engagement < 0.5:  # Low engagement
                # Change content types or add interactive elements
                adaptations_made.append("Added interactive content")
                self._add_engaging_content(path)
            
            # Adapt based on time efficiency
            if time_efficiency < 0.7:  # Taking too long
                # Simplify content or provide additional support
                adaptations_made.append("Simplified content structure")
                self._simplify_content(path)
            elif time_efficiency > 1.5:  # Too fast
                # Add challenging content or deeper exploration
                adaptations_made.append("Added challenging content")
                self._add_challenging_content(path)
            
            # Update path metadata
            path.last_updated = datetime.now()
            if adaptations_made:
                self.logger.info(f"Adapted learning path {path_id}: {', '.join(adaptations_made)}")
            
            return path
        except Exception as e:
            self.logger.error(f"Failed to adapt learning path: {e}")
            raise
    
    def _sequence_content(self, content_list: List[LearningContent], learner: LearnerProfile) -> List[LearningContent]:
        """Sequence content based on difficulty and prerequisites."""
        # Simple sequencing by difficulty level
        difficulty_order = [DifficultyLevel.BEGINNER, DifficultyLevel.ELEMENTARY,
                          DifficultyLevel.INTERMEDIATE, DifficultyLevel.ADVANCED, DifficultyLevel.EXPERT]
        
        sequenced = sorted(content_list, key=lambda x: difficulty_order.index(x.difficulty_level))
        
        # TODO: Implement more sophisticated prerequisite-based sequencing
        return sequenced
    
    def _create_milestones(self, content_sequence: List[LearningContent]) -> List[Dict[str, Any]]:
        """Create learning milestones."""
        milestones = []
        total_content = len(content_sequence)
        
        # Create milestones at 25%, 50%, 75%, and 100%
        milestone_points = [0.25, 0.5, 0.75, 1.0]
        
        for i, point in enumerate(milestone_points):
            content_index = int(total_content * point) - 1
            if content_index >= 0 and content_index < total_content:
                milestone = {
                    'milestone_id': f"M{i+1}",
                    'name': f"Milestone {i+1}",
                    'content_index': content_index,
                    'progress_percentage': point * 100,
                    'description': f"Complete {point*100:.0f}% of learning path",
                    'achieved': False
                }
                milestones.append(milestone)
        
        return milestones
    
    def _add_remedial_content(self, path: LearningPath) -> None:
        """Add remedial content to learning path."""
        # Placeholder for adding easier content
        pass
    
    def _increase_difficulty(self, path: LearningPath) -> None:
        """Increase difficulty of learning path."""
        # Placeholder for increasing difficulty
        pass
    
    def _add_engaging_content(self, path: LearningPath) -> None:
        """Add more engaging content types."""
        # Placeholder for adding interactive/engaging content
        pass
    
    def _simplify_content(self, path: LearningPath) -> None:
        """Simplify content structure."""
        # Placeholder for content simplification
        pass
    
    def _add_challenging_content(self, path: LearningPath) -> None:
        """Add more challenging content."""
        # Placeholder for adding challenging content
        pass


class IntelligentTutorSystem:
    """Provides intelligent tutoring and real-time assistance."""
    
    def __init__(self, config: EducationalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tutoring_sessions = {}
        self.help_patterns = {}
        self.feedback_templates = {}
    
    def start_tutoring_session(self, learner: LearnerProfile, content: LearningContent) -> str:
        """Start an intelligent tutoring session."""
        try:
            session_id = f"TUTOR_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{learner.learner_id}"
            
            session = {
                'session_id': session_id,
                'learner_id': learner.learner_id,
                'content_id': content.content_id,
                'start_time': datetime.now(),
                'interactions': [],
                'current_step': 0,
                'help_requests': 0,
                'hints_given': 0,
                'mistakes_made': 0,
                'progress': 0.0
            }
            
            self.tutoring_sessions[session_id] = session
            
            # Generate initial guidance
            initial_guidance = self._generate_initial_guidance(learner, content)
            session['interactions'].append({
                'timestamp': datetime.now(),
                'type': 'guidance',
                'content': initial_guidance
            })
            
            self.logger.info(f"Started tutoring session {session_id} for {learner.learner_id}")
            return session_id
        except Exception as e:
            self.logger.error(f"Failed to start tutoring session: {e}")
            raise
    
    def provide_help(self, session_id: str, question: str) -> str:
        """Provide contextual help during tutoring."""
        try:
            if session_id not in self.tutoring_sessions:
                return "Session not found. Please start a new tutoring session."
            
            session = self.tutoring_sessions[session_id]
            session['help_requests'] += 1
            
            # Analyze the question and provide appropriate help
            help_response = self._analyze_help_request(question, session)
            
            # Record interaction
            session['interactions'].append({
                'timestamp': datetime.now(),
                'type': 'help_request',
                'question': question,
                'response': help_response
            })
            
            self.logger.info(f"Provided help in session {session_id}")
            return help_response
        except Exception as e:
            self.logger.error(f"Failed to provide help: {e}")
            return "I'm sorry, I encountered an error while trying to help. Please try again."
    
    def provide_feedback(self, session_id: str, learner_response: str, correct_answer: str) -> Dict[str, Any]:
        """Provide intelligent feedback on learner responses."""
        try:
            if session_id not in self.tutoring_sessions:
                return {'error': 'Session not found'}
            
            session = self.tutoring_sessions[session_id]
            
            # Analyze the response
            is_correct = self._evaluate_response(learner_response, correct_answer)
            feedback = self._generate_feedback(learner_response, correct_answer, is_correct)
            
            if not is_correct:
                session['mistakes_made'] += 1
                # Provide hint if needed
                if session['mistakes_made'] % 2 == 0:  # Every second mistake
                    hint = self._generate_hint(correct_answer, session)
                    feedback['hint'] = hint
                    session['hints_given'] += 1
            
            # Update progress
            session['current_step'] += 1
            session['progress'] = min(1.0, session['current_step'] / 10)  # Assume 10 steps total
            
            # Record interaction
            session['interactions'].append({
                'timestamp': datetime.now(),
                'type': 'feedback',
                'learner_response': learner_response,
                'correct_answer': correct_answer,
                'is_correct': is_correct,
                'feedback': feedback
            })
            
            return feedback
        except Exception as e:
            self.logger.error(f"Failed to provide feedback: {e}")
            return {'error': str(e)}
    
    def _generate_initial_guidance(self, learner: LearnerProfile, content: LearningContent) -> str:
        """Generate initial guidance for the tutoring session."""
        guidance = f"Welcome {learner.name}! Let's work on {content.title} together. "
        
        if content.difficulty_level == DifficultyLevel.BEGINNER:
            guidance += "We'll start with the basics and build up your understanding step by step."
        elif content.difficulty_level == DifficultyLevel.ADVANCED:
            guidance += "This is advanced material, so take your time and don't hesitate to ask for help."
        else:
            guidance += "Let's explore this topic together. Feel free to ask questions anytime."
        
        return guidance
    
    def _analyze_help_request(self, question: str, session: Dict[str, Any]) -> str:
        """Analyze help request and provide appropriate response."""
        question_lower = question.lower()
        
        # Common help patterns
        if any(word in question_lower for word in ['what', 'define', 'meaning']):
            return "Let me explain the key concepts. [Concept explanation would be generated here based on current content]"
        elif any(word in question_lower for word in ['how', 'steps', 'process']):
            return "Here's a step-by-step approach: [Step-by-step guidance would be generated here]"
        elif any(word in question_lower for word in ['why', 'reason', 'because']):
            return "The reasoning behind this is: [Explanation of reasoning would be provided here]"
        elif any(word in question_lower for word in ['example', 'sample', 'instance']):
            return "Here's a helpful example: [Relevant example would be provided here]"
        else:
            return "I understand you need help. Can you be more specific about what you're struggling with?"
    
    def _evaluate_response(self, learner_response: str, correct_answer: str) -> bool:
        """Evaluate if learner response is correct."""
        # Simplified evaluation (in real implementation, would use NLP and domain-specific logic)
        learner_clean = learner_response.strip().lower()
        correct_clean = correct_answer.strip().lower()
        
        # Exact match
        if learner_clean == correct_clean:
            return True
        
        # Partial match (contains key terms)
        correct_words = set(correct_clean.split())
        learner_words = set(learner_clean.split())
        overlap = len(correct_words.intersection(learner_words))
        
        # Consider correct if significant overlap
        return overlap >= len(correct_words) * 0.7
    
    def _generate_feedback(self, learner_response: str, correct_answer: str, is_correct: bool) -> Dict[str, Any]:
        """Generate personalized feedback."""
        feedback = {
            'is_correct': is_correct,
            'message': '',
            'encouragement': '',
            'next_steps': ''
        }
        
        if is_correct:
            feedback['message'] = "Excellent! That's correct."
            feedback['encouragement'] = "You're doing great! Keep up the good work."
            feedback['next_steps'] = "Let's move on to the next concept."
        else:
            feedback['message'] = "Not quite right, but you're on the right track."
            feedback['encouragement'] = "Don't worry, learning takes practice. Let's try again."
            feedback['next_steps'] = "Think about the key concepts we discussed earlier."
        
        return feedback
    
    def _generate_hint(self, correct_answer: str, session: Dict[str, Any]) -> str:
        """Generate a helpful hint."""
        # Simplified hint generation
        hints = [
            "Think about the main concept we just covered.",
            "Remember the key terms from the lesson.",
            "Consider the examples we looked at earlier.",
            "Break the problem down into smaller parts.",
            "What patterns do you notice?"
        ]
        
        # Return hint based on number of hints already given
        hint_index = session['hints_given'] % len(hints)
        return hints[hint_index]


class LearningAnalytics:
    """Provides learning analytics and insights."""
    
    def __init__(self, config: EducationalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.analytics_data = {}
        self.reports = {}
    
    def analyze_learner_progress(self, learner_id: str, sessions: List[LearningSession], 
                               assessments: List[AssessmentResult]) -> Dict[str, Any]:
        """Analyze comprehensive learner progress."""
        try:
            if not sessions and not assessments:
                return {'error': 'No data available for analysis'}
            
            analysis = {
                'learner_id': learner_id,
                'analysis_date': datetime.now().isoformat(),
                'session_analytics': {},
                'assessment_analytics': {},
                'overall_progress': {},
                'recommendations': []
            }
            
            # Session analytics
            if sessions:
                analysis['session_analytics'] = self._analyze_sessions(sessions)
            
            # Assessment analytics
            if assessments:
                analysis['assessment_analytics'] = self._analyze_assessments(assessments)
            
            # Overall progress
            analysis['overall_progress'] = self._calculate_overall_progress(sessions, assessments)
            
            # Generate recommendations
            analysis['recommendations'] = self._generate_recommendations(analysis)
            
            # Store analytics
            self.analytics_data[learner_id] = analysis
            
            self.logger.info(f"Analyzed progress for learner {learner_id}")
            return analysis
        except Exception as e:
            self.logger.error(f"Failed to analyze learner progress: {e}")
            return {'error': str(e)}
    
    def _analyze_sessions(self, sessions: List[LearningSession]) -> Dict[str, Any]:
        """Analyze learning sessions."""
        if not sessions:
            return {}
        
        total_sessions = len(sessions)
        completed_sessions = sum(1 for s in sessions if s.completed)
        total_time = sum(s.time_on_task_minutes for s in sessions)
        total_activities = sum(s.activities_completed for s in sessions)
        
        # Calculate averages
        avg_engagement = statistics.mean([s.engagement_score for s in sessions]) if sessions else 0
        avg_accuracy = 0
        if any(s.total_questions > 0 for s in sessions):
            accuracy_scores = [s.correct_answers / s.total_questions for s in sessions if s.total_questions > 0]
            avg_accuracy = statistics.mean(accuracy_scores) if accuracy_scores else 0
        
        # Engagement trend
        engagement_scores = [s.engagement_score for s in sessions[-10:]]  # Last 10 sessions
        engagement_trend = 'stable'
        if len(engagement_scores) >= 3:
            if engagement_scores[-1] > engagement_scores[0] + 0.1:
                engagement_trend = 'improving'
            elif engagement_scores[-1] < engagement_scores[0] - 0.1:
                engagement_trend = 'declining'
        
        return {
            'total_sessions': total_sessions,
            'completed_sessions': completed_sessions,
            'completion_rate': completed_sessions / total_sessions if total_sessions > 0 else 0,
            'total_study_time_minutes': total_time,
            'average_session_time': total_time / total_sessions if total_sessions > 0 else 0,
            'total_activities_completed': total_activities,
            'average_engagement': avg_engagement,
            'average_accuracy': avg_accuracy,
            'engagement_trend': engagement_trend
        }
    
    def _analyze_assessments(self, assessments: List[AssessmentResult]) -> Dict[str, Any]:
        """Analyze assessment results."""
        if not assessments:
            return {}
        
        total_assessments = len(assessments)
        scores = [a.percentage for a in assessments]
        
        avg_score = statistics.mean(scores)
        highest_score = max(scores)
        lowest_score = min(scores)
        
        # Performance trend
        recent_scores = scores[-5:]  # Last 5 assessments
        performance_trend = 'stable'
        if len(recent_scores) >= 3:
            if recent_scores[-1] > recent_scores[0] + 10:  # 10% improvement
                performance_trend = 'improving'
            elif recent_scores[-1] < recent_scores[0] - 10:  # 10% decline
                performance_trend = 'declining'
        
        # Identify strengths and weaknesses
        all_strengths = []
        all_weaknesses = []
        for assessment in assessments:
            all_strengths.extend(assessment.strengths_identified)
            all_weaknesses.extend(assessment.areas_for_improvement)
        
        # Count frequency of strengths and weaknesses
        strength_counts = {}
        weakness_counts = {}
        
        for strength in all_strengths:
            strength_counts[strength] = strength_counts.get(strength, 0) + 1
        
        for weakness in all_weaknesses:
            weakness_counts[weakness] = weakness_counts.get(weakness, 0) + 1
        
        # Top strengths and weaknesses
        top_strengths = sorted(strength_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_weaknesses = sorted(weakness_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'total_assessments': total_assessments,
            'average_score': avg_score,
            'highest_score': highest_score,
            'lowest_score': lowest_score,
            'score_range': highest_score - lowest_score,
            'performance_trend': performance_trend,
            'top_strengths': [strength for strength, count in top_strengths],
            'top_weaknesses': [weakness for weakness, count in top_weaknesses]
        }
    
    def _calculate_overall_progress(self, sessions: List[LearningSession], 
                                  assessments: List[AssessmentResult]) -> Dict[str, Any]:
        """Calculate overall learning progress."""
        progress = {
            'overall_score': 0.0,
            'engagement_level': EngagementLevel.MODERATE,
            'learning_velocity': 'normal',
            'mastery_level': 'developing'
        }
        
        # Calculate overall score (weighted average of sessions and assessments)
        session_score = 0.0
        assessment_score = 0.0
        
        if sessions:
            engagement_scores = [s.engagement_score for s in sessions]
            session_score = statistics.mean(engagement_scores) if engagement_scores else 0
        
        if assessments:
            assessment_scores = [a.percentage / 100 for a in assessments]
            assessment_score = statistics.mean(assessment_scores) if assessment_scores else 0
        
        # Weighted combination (60% assessments, 40% engagement)
        if assessments and sessions:
            progress['overall_score'] = 0.6 * assessment_score + 0.4 * session_score
        elif assessments:
            progress['overall_score'] = assessment_score
        elif sessions:
            progress['overall_score'] = session_score
        
        # Determine engagement level
        if session_score >= 0.8:
            progress['engagement_level'] = EngagementLevel.VERY_HIGH
        elif session_score >= 0.6:
            progress['engagement_level'] = EngagementLevel.HIGH
        elif session_score >= 0.4:
            progress['engagement_level'] = EngagementLevel.MODERATE
        elif session_score >= 0.2:
            progress['engagement_level'] = EngagementLevel.LOW
        else:
            progress['engagement_level'] = EngagementLevel.VERY_LOW
        
        # Determine mastery level
        if progress['overall_score'] >= 0.9:
            progress['mastery_level'] = 'expert'
        elif progress['overall_score'] >= 0.8:
            progress['mastery_level'] = 'proficient'
        elif progress['overall_score'] >= 0.6:
            progress['mastery_level'] = 'developing'
        else:
            progress['mastery_level'] = 'beginning'
        
        return progress
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate personalized recommendations."""
        recommendations = []
        
        overall_progress = analysis.get('overall_progress', {})
        session_analytics = analysis.get('session_analytics', {})
        assessment_analytics = analysis.get('assessment_analytics', {})
        
        # Engagement recommendations
        engagement_level = overall_progress.get('engagement_level', EngagementLevel.MODERATE)
        if engagement_level in [EngagementLevel.LOW, EngagementLevel.VERY_LOW]:
            recommendations.append("Try more interactive and gamified content to boost engagement")
            recommendations.append("Consider shorter study sessions with more frequent breaks")
        
        # Performance recommendations
        avg_score = assessment_analytics.get('average_score', 0)
        if avg_score < 60:
            recommendations.append("Focus on foundational concepts before moving to advanced topics")
            recommendations.append("Consider additional practice exercises and review sessions")
        elif avg_score > 90:
            recommendations.append("Ready for more challenging content and advanced topics")
            recommendations.append("Consider peer tutoring or project-based learning")
        
        # Study habits recommendations
        completion_rate = session_analytics.get('completion_rate', 0)
        if completion_rate < 0.7:
            recommendations.append("Work on completing learning sessions to build consistency")
            recommendations.append("Set specific study goals and track progress daily")
        
        # Trend-based recommendations
        performance_trend = assessment_analytics.get('performance_trend', 'stable')
        if performance_trend == 'declining':
            recommendations.append("Review recent topics to identify knowledge gaps")
            recommendations.append("Consider seeking additional support or tutoring")
        elif performance_trend == 'improving':
            recommendations.append("Great progress! Continue with current study strategies")
        
        return recommendations


class EducationalAISystem:
    """Main educational AI system integrating all components."""
    
    def __init__(self, config: Optional[EducationalConfig] = None):
        self.config = config or EducationalConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.content_recommender = ContentRecommendationEngine(self.config)
        self.adaptive_engine = AdaptiveLearningEngine(self.config)
        self.tutor_system = IntelligentTutorSystem(self.config)
        self.analytics = LearningAnalytics(self.config)
        
        # Data storage
        self.learners = {}
        self.content_library = {}
        self.learning_sessions = {}
        self.assessments = {}
        self.assessment_results = {}
        
        self.running = False
    
    async def start(self) -> None:
        """Start educational AI system."""
        try:
            self.running = True
            self.logger.info(f"Educational AI system started for institution {self.config.institution_id}")
        except Exception as e:
            self.logger.error(f"Failed to start educational AI system: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop educational AI system."""
        try:
            self.running = False
            self.logger.info("Educational AI system stopped")
        except Exception as e:
            self.logger.error(f"Failed to stop educational AI system: {e}")
            raise
    
    def register_learner(self, learner: LearnerProfile) -> None:
        """Register a new learner."""
        try:
            self.learners[learner.learner_id] = learner
            self.logger.info(f"Registered learner: {learner.name} ({learner.learner_id})")
        except Exception as e:
            self.logger.error(f"Failed to register learner: {e}")
            raise
    
    def add_content(self, content: LearningContent) -> None:
        """Add content to the system."""
        try:
            self.content_library[content.content_id] = content
            self.content_recommender.add_content(content)
            self.logger.info(f"Added content: {content.title} ({content.content_id})")
        except Exception as e:
            self.logger.error(f"Failed to add content: {e}")
            raise
    
    def get_personalized_recommendations(self, learner_id: str, subject: str) -> List[LearningContent]:
        """Get personalized content recommendations."""
        try:
            if learner_id not in self.learners:
                raise ValueError(f"Learner {learner_id} not found")
            
            learner = self.learners[learner_id]
            recommendations = self.content_recommender.recommend_content(learner, subject)
            
            return recommendations
        except Exception as e:
            self.logger.error(f"Failed to get recommendations: {e}")
            return []
    
    def create_adaptive_path(self, learner_id: str, subject: str, goal: str) -> LearningPath:
        """Create an adaptive learning path."""
        try:
            if learner_id not in self.learners:
                raise ValueError(f"Learner {learner_id} not found")
            
            learner = self.learners[learner_id]
            path = self.adaptive_engine.create_learning_path(learner, subject, goal, self.content_library)
            
            return path
        except Exception as e:
            self.logger.error(f"Failed to create adaptive path: {e}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get educational AI system status."""
        try:
            status = {
                'institution_id': self.config.institution_id,
                'system_running': self.running,
                'adaptive_learning_enabled': self.config.adaptive_learning_enabled,
                'personalization_enabled': self.config.personalization_enabled,
                'total_learners': len(self.learners),
                'total_content_items': len(self.content_library),
                'active_learning_sessions': len([s for s in self.learning_sessions.values() if not s.completed]),
                'total_assessments': len(self.assessments),
                'timestamp': datetime.now().isoformat()
            }
            
            return status
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {}


# Global educational AI system instance
_educational_system: Optional[EducationalAISystem] = None


def initialize_educational(config: Optional[EducationalConfig] = None) -> None:
    """Initialize educational AI system."""
    global _educational_system
    _educational_system = EducationalAISystem(config)


async def shutdown_educational() -> None:
    """Shutdown educational AI system."""
    global _educational_system
    if _educational_system:
        await _educational_system.stop()
        _educational_system = None