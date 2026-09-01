from __future__ import annotations

import os
import unittest
from unittest.mock import patch


os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

try:
    import pygame
except ModuleNotFoundError:
    pygame = None  # type: ignore[assignment]
else:
    from multi_agent_sim.controllers import create_default_controller_registry
    from multi_agent_sim.visualization import (
        PygameSimulationApp,
        run_pygame_application,
    )


class _RaisingController:
    def choose_action(self, observation: object, robot_id: str) -> object:
        raise RuntimeError(f"test controller failed for {robot_id}")


@unittest.skipUnless(pygame is not None, "pygame visualization extra is not installed")
class PygameApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        registry = create_default_controller_registry()
        registry.register(
            "raising",
            "Raising test controller",
            lambda context: _RaisingController(),
        )
        self.app = PygameSimulationApp(
            width=6,
            height=5,
            num_robots=3,
            num_items=4,
            seed=42,
            max_steps=12,
            step_rate=5,
            controller_registry=registry,
        )
        self.app.initialize()
        self.app.render()

    def tearDown(self) -> None:
        pygame.quit()

    @staticmethod
    def _click(position: tuple[int, int]) -> pygame.event.Event:
        return pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=position,
        )

    def test_manual_mode_selects_and_places_a_robot_from_grid_click(self) -> None:
        self.app.process_event(self._click(self.app._manual_button.rect.center))
        self.assertEqual(self.app.setup.robot_mode, "manual")
        self.assertTrue(self.app.setup_is_valid)

        self.app.setup.selected_robot = 0
        grid = self.app._preview_grid
        cell = (4, 4)
        click_position = (
            round(grid.x + (cell[0] + 0.5) * grid.width / 6),
            round(grid.y + (cell[1] + 0.5) * grid.height / 5),
        )
        self.app.process_event(self._click(click_position))

        draft = self.app.setup.robot_drafts[0]
        self.assertEqual((draft.x.text, draft.y.text), ("4", "4"))
        self.app.render()

    def test_toolbar_events_pause_navigation_and_timed_playback(self) -> None:
        self.app.process_event(self._click(self.app._start_button.rect.center))
        self.assertEqual(self.app.screen_name, "simulation")
        session = self.app.session
        assert session is not None
        self.assertFalse(session.playing)
        self.assertEqual(session.cursor, 0)

        self.app.process_event(self._click(self.app._play_button.rect.center))
        self.assertTrue(session.playing)
        self.app.update(0.41)
        self.assertEqual(session.cursor, 2)

        # Space is routed only once when the Play button owns keyboard focus.
        self.app.process_event(
            pygame.event.Event(
                pygame.KEYDOWN,
                key=pygame.K_SPACE,
                mod=0,
                unicode=" ",
            )
        )
        self.assertFalse(session.playing)

        self.app.process_event(self._click(self.app._back_button.rect.center))
        self.assertFalse(session.playing)
        self.assertEqual(session.cursor, 1)
        self.app.process_event(self._click(self.app._forward_button.rect.center))
        self.assertEqual(session.cursor, 2)

        # A button that becomes disabled while focused must not trap global
        # shortcuts. Back twice reaches step zero and leaves Back focused.
        self.app.process_event(self._click(self.app._back_button.rect.center))
        self.app.process_event(self._click(self.app._back_button.rect.center))
        self.assertEqual(session.cursor, 0)
        self.app.process_event(
            pygame.event.Event(
                pygame.KEYDOWN,
                key=pygame.K_SPACE,
                mod=0,
                unicode=" ",
            )
        )
        self.assertTrue(session.playing)
        session.pause()

        self.app.process_event(self._click(self.app._new_setup_button.rect.center))
        self.assertEqual(self.app.screen_name, "setup")
        self.assertIsNone(self.app.session)
        self.assertIsNone(self.app._last_world)
        self.assertTrue(self.app.setup_is_valid)

    def test_slider_arrow_changes_only_the_rate_when_it_has_focus(self) -> None:
        self.app.process_event(self._click(self.app._start_button.rect.center))
        session = self.app.session
        assert session is not None
        session.step_forward()
        cursor = session.cursor

        self.app.process_event(self._click(self.app._rate_slider.rect.center))
        starting_rate = session.step_rate
        self.app.process_event(
            pygame.event.Event(
                pygame.KEYDOWN,
                key=pygame.K_LEFT,
                mod=0,
                unicode="",
            )
        )

        self.assertEqual(session.cursor, cursor)
        self.assertEqual(session.step_rate, max(1, starting_rate - 1))

    def test_clipped_manual_rows_do_not_intercept_setup_field_clicks(self) -> None:
        self.app.setup.fields["num_robots"].set_text("20")
        self.app.setup.sync_robot_drafts()
        self.app.setup.set_robot_mode("manual")
        self.app.setup.scroll_offset = 10_000
        self.app._layout_setup()

        robot_x_values = tuple(
            draft.x.text for draft in self.app.setup.robot_drafts
        )
        width_field = self.app.setup.fields["width"]
        self.app.process_event(self._click(width_field.rect.center))
        self.app.process_event(
            pygame.event.Event(
                pygame.KEYDOWN,
                key=pygame.K_9,
                mod=0,
                unicode="9",
            )
        )

        self.assertEqual(width_field.text, "69")
        self.assertEqual(
            tuple(draft.x.text for draft in self.app.setup.robot_drafts),
            robot_x_values,
        )

    def test_shrinking_manual_robot_count_clamps_scroll_position(self) -> None:
        self.app.setup.fields["num_robots"].set_text("20")
        self.app.setup.sync_robot_drafts()
        self.app.setup.set_robot_mode("manual")
        self.app.setup.scroll_offset = 10_000
        self.app._layout_setup()
        self.assertGreater(self.app.setup.scroll_offset, 0)

        self.app.setup.fields["num_robots"].set_text("2")
        self.app.setup.sync_robot_drafts()
        self.app._layout_setup()

        self.assertEqual(self.app.setup.scroll_offset, 0)
        self.assertEqual(len(self.app._manual_row_rects), 2)
        self.assertTrue(
            all(
                row.colliderect(self.app._manual_view)
                for row in self.app._manual_row_rects
            )
        )

    def test_cost_fields_validate_and_reach_the_session_world(self) -> None:
        movement = self.app.setup.fields["movement_cost"]
        pickup = self.app.setup.fields["pickup_cost"]
        drop = self.app.setup.fields["drop_cost"]
        wait = self.app.setup.fields["wait_cost"]

        movement.set_text("-0.25")
        self.assertFalse(self.app.setup_is_valid)
        self.assertIn("movement_cost", self.app.setup.validate().field_errors)
        movement.set_text("nan")
        self.assertFalse(self.app.setup_is_valid)

        movement.set_text("2.5")
        pickup.set_text("3")
        drop.set_text("inf")
        self.assertFalse(self.app.setup_is_valid)
        self.assertIn("drop_cost", self.app.setup.validate().field_errors)

        drop.set_text("4.5")
        wait.set_text("0.25")
        self.assertTrue(self.app.setup_is_valid)
        self.app.process_event(self._click(self.app._start_button.rect.center))

        session = self.app.session
        assert session is not None
        costs = session.world.action_battery_costs
        self.assertEqual(
            (costs.movement, costs.pickup, costs.wait, costs.drop),
            (2.5, 3.0, 0.25, 4.5),
        )

        self.app.process_event(self._click(self.app._new_setup_button.rect.center))
        self.assertEqual(self.app.screen_name, "setup")
        self.assertEqual(self.app.setup.fields["drop_cost"].text, "4.5")

    def test_drop_cost_constructor_and_wrapper_prefill_the_setup(self) -> None:
        app = PygameSimulationApp(
            width=2,
            height=2,
            num_robots=1,
            num_items=1,
            seed=3,
            max_steps=4,
            step_rate=5,
            drop_cost=2.75,
        )
        self.assertEqual(app.setup.fields["drop_cost"].text, "2.75")
        configuration = app.setup.validate().configuration
        assert configuration is not None
        self.assertEqual(configuration.action_battery_costs.drop, 2.75)

        with patch(
            "multi_agent_sim.visualization.pygame_app.PygameSimulationApp"
        ) as app_type:
            result = run_pygame_application(
                width=2,
                height=2,
                num_robots=1,
                num_items=1,
                seed=3,
                max_steps=4,
                step_rate=5,
                drop_cost=3.25,
            )

        self.assertIs(result, app_type.return_value.run.return_value)
        self.assertEqual(app_type.call_args.kwargs["drop_cost"], 3.25)

    def test_four_cost_fields_fit_at_the_minimum_window_size(self) -> None:
        self.app.process_event(
            pygame.event.Event(
                pygame.VIDEORESIZE,
                w=self.app._MIN_WINDOW_SIZE[0],
                h=self.app._MIN_WINDOW_SIZE[1],
            )
        )
        self.app.render()

        cost_fields = tuple(
            self.app.setup.fields[name].rect
            for name in (
                "movement_cost",
                "pickup_cost",
                "drop_cost",
                "wait_cost",
            )
        )
        self.assertTrue(
            all(self.app._left_panel.contains(rect) for rect in cost_fields)
        )
        self.assertTrue(all(rect.width >= 80 for rect in cost_fields))
        self.assertTrue(
            all(
                not first.colliderect(second)
                for index, first in enumerate(cost_fields)
                for second in cost_fields[index + 1 :]
            )
        )

    def test_controller_choices_survive_reroll_mode_and_count_changes(self) -> None:
        selectors = self.app.setup.controller_selectors
        self.assertEqual(
            selectors[0].choices,
            (
                ("random", "Random"),
                ("nearest_item", "Nearest Item"),
                ("raising", "Raising test controller"),
            ),
        )
        selectors[2].value = "raising"
        self.app.setup.reroll()
        self.app.setup.set_robot_mode("manual")

        self.app.setup.fields["num_robots"].set_text("1")
        self.app.setup.sync_robot_drafts()
        self.app.setup.fields["num_robots"].set_text("3")
        self.app.setup.sync_robot_drafts()
        for index, draft in enumerate(self.app.setup.robot_drafts):
            draft.x.set_text(str(index))
            draft.y.set_text("0")

        self.assertEqual(self.app.setup.controller_selectors[2].value, "raising")
        configuration = self.app.setup.validate().configuration
        assert configuration is not None
        self.assertEqual(configuration.controller_keys[2], "raising")
        assert configuration.robot_configurations is not None
        self.assertEqual(
            configuration.robot_configurations[2].controller_key,
            "raising",
        )

    def test_custom_controller_warning_and_scrollable_status_panel(self) -> None:
        self.app.setup.fields["num_robots"].set_text("10")
        self.app.setup.sync_robot_drafts()
        self.app.setup.controller_selectors[0].value = "raising"
        self.app.setup.invalidate_preview()
        self.app.render()
        self.app.process_event(self._click(self.app._start_button.rect.center))

        session = self.app.session
        assert session is not None
        self.app.process_event(self._click(self.app._play_button.rect.center))
        self.app.update(0.21)
        self.assertTrue(session.playing)
        self.assertEqual(session.cursor, 1)
        self.assertIn("test controller failed", session.controller_errors["robot_1"])
        self.assertEqual(session.controller_key_for("robot_1"), "raising")
        self.assertEqual(
            session.controller_display_name_for("robot_1"),
            "Raising test controller",
        )
        robot = session.world.get_entity("robot_1")
        self.assertEqual(robot.battery_level, 100.0)
        self.assertIsNone(robot.carried_item_id)

        self.app.update(0.21)
        self.assertTrue(session.playing)
        self.assertEqual(session.cursor, 2)
        self.assertIn("test controller failed", session.controller_errors["robot_1"])

        self.app.render()
        self.assertLess(
            self.app._simulation_grid_area.right,
            self.app._status_panel.x,
        )
        self.assertEqual(self.app._status_scroll_offset, 0)
        self.app.process_event(
            pygame.event.Event(
                pygame.MOUSEWHEEL,
                y=-1,
                x=0,
                pos=self.app._status_view.center,
            )
        )
        self.assertGreater(self.app._status_scroll_offset, 0)
        self.app.render()

    def test_status_number_format_is_compact_and_never_ends_in_a_dot(self) -> None:
        self.assertEqual(self.app._format_status_number(94.5), "94.5")
        self.assertEqual(self.app._format_status_number(0.001), "0.001")
        self.assertFalse(self.app._format_status_number(1.00001).endswith("."))


if __name__ == "__main__":
    unittest.main()
